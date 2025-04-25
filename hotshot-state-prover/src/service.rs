//! A light client prover service

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::Arc,
    thread::LocalKey,
    time::{Duration, Instant},
};

use alloy::{
    network::EthereumWallet,
    primitives::{Address, U256},
    providers::{Provider, ProviderBuilder},
    rpc::types::TransactionReceipt,
    signers::{k256::ecdsa::SigningKey, local::LocalSigner},
};
use anyhow::{anyhow, Context, Result};
use displaydoc::Display;
use espresso_types::{config::PublicNetworkConfig, SeqTypes};
use futures::FutureExt;
use hotshot_contract_adapter::{
    field_to_u256,
    sol_types::{LightClientStateSol, LightClientV2, PlonkProofSol, StakeTableStateSol},
};
use hotshot_query_service::availability::StateCertQueryData;
use hotshot_types::{
    data::EpochNumber,
    light_client::{
        compute_stake_table_commitment, CircuitField, LightClientState, PublicInput,
        StakeTableState, StateSignature, StateSignaturesBundle, StateVerKey,
    },
    simple_certificate::LightClientStateUpdateCertificate,
    traits::{
        node_implementation::{ConsensusTime, NodeType},
        signature_key::StateSignatureKey,
        stake_table::StakeTableError,
    },
    utils::{
        epoch_from_block_number, is_epoch_root, is_ge_epoch_root, option_epoch_from_block_number,
    },
    PeerConfig,
};
use jf_pcs::prelude::UnivariateUniversalParams;
use jf_plonk::errors::PlonkError;
use jf_relation::Circuit as _;
use sequencer_utils::deployer::is_proxy_contract;
use surf_disco::Client;
use tide_disco::{error::ServerError, Api};
use time::ext::InstantExt;
use tokio::{io, spawn, task::spawn_blocking, time::sleep};
use url::Url;
use vbs::version::{StaticVersion, StaticVersionType};

use crate::snark::{generate_state_update_proof, Proof, ProvingKey};

// Thread-local storage for the current prover state to enable accessing it from any function
thread_local! {
    pub static CURRENT_PROVER_STATE: RefCell<Option<ProverServiceState>> = RefCell::new(None);
}

/// Configuration/Parameters used for hotshot state prover
#[derive(Debug, Clone)]
pub struct StateProverConfig {
    /// Url of the state relay server (a CDN that sequencers push their Schnorr signatures to)
    pub relay_server: Url,
    /// Interval between light client state update
    pub update_interval: Duration,
    /// Interval between retries if a state update fails
    pub retry_interval: Duration,
    /// URL of the chain (layer 1  or any layer 2) JSON-RPC provider.
    pub provider_endpoint: Url,
    /// Address of LightClient proxy contract (primary contract)
    pub light_client_address: Address,
    /// Additional addresses of LightClient proxy contracts to submit proofs to
    pub additional_light_client_addresses: Vec<Address>,
    /// Transaction signing key for Ethereum or any other layer 2
    pub signer: LocalSigner<SigningKey>,
    /// URL of a node that is currently providing the HotShot config.
    /// This is used to initialize the stake table.
    pub sequencer_url: Url,
    /// If daemon and provided, the service will run a basic HTTP server on the given port.
    ///
    /// The server provides healthcheck and version endpoints.
    pub port: Option<u16>,
    /// Stake table capacity for the prover circuit.
    pub stake_table_capacity: usize,
    /// Epoch length in number of Hotshot blocks.
    pub blocks_per_epoch: u64,
    /// The epoch start block.
    pub epoch_start_block: u64,
    /// Maximum number of retires for one-shot prover
    pub max_retries: u64,
}

#[derive(Debug, Clone)]
pub struct ProverServiceState {
    /// The configuration of the prover service
    pub config: StateProverConfig,
    /// The current epoch number of the stake table
    pub epoch: Option<<SeqTypes as NodeType>::Epoch>,
    /// The stake table
    pub stake_table: Vec<PeerConfig<SeqTypes>>,
    /// The current stake table state
    pub st_state: StakeTableState,
}

impl ProverServiceState {
    pub async fn new_genesis(config: StateProverConfig) -> Result<Self> {
        let stake_table = fetch_stake_table_from_sequencer(&config.sequencer_url, None)
            .await
            .with_context(|| "Failed to initialize stake table")?;
        let st_state = compute_stake_table_commitment(&stake_table, config.stake_table_capacity);
        Ok(Self {
            config,
            epoch: None,
            stake_table,
            st_state,
        })
    }

    pub async fn sync_with_epoch(
        &mut self,
        epoch: Option<<SeqTypes as NodeType>::Epoch>,
    ) -> Result<()> {
        if epoch != self.epoch {
            self.stake_table = fetch_stake_table_from_sequencer(&self.config.sequencer_url, epoch)
                .await
                .with_context(|| format!("Failed to update stake table for epoch: {:?}", epoch))?;
            self.st_state =
                compute_stake_table_commitment(&self.stake_table, self.config.stake_table_capacity);
            self.epoch = epoch;
        }
        Ok(())
    }
}

impl StateProverConfig {
    pub async fn validate_light_client_contract(&self) -> anyhow::Result<()> {
        let provider = ProviderBuilder::new().on_http(self.provider_endpoint.clone());

        // Validate primary contract
        if !is_proxy_contract(&provider, self.light_client_address).await? {
            anyhow::bail!(
                "Light Client contract's address {:?} is not a proxy",
                self.light_client_address
            );
        }

        // Validate additional contracts
        for (index, address) in self.additional_light_client_addresses.iter().enumerate() {
            if !is_proxy_contract(&provider, *address).await? {
                anyhow::bail!(
                    "Additional Light Client contract's address {:?} at index {} is not a proxy",
                    address,
                    index
                );
            }
        }

        Ok(())
    }
}

/// Get the epoch-related  from the sequencer's `PublicHotShotConfig` struct
/// return (blocks_per_epoch, epoch_start_block)
pub async fn fetch_epoch_config_from_sequencer(sequencer_url: &Url) -> anyhow::Result<(u64, u64)> {
    // Request the configuration until it is successful
    let epoch_config = loop {
        match surf_disco::Client::<tide_disco::error::ServerError, StaticVersion<0, 1>>::new(
            sequencer_url.clone(),
        )
        .get::<PublicNetworkConfig>("config/hotshot")
        .send()
        .await
        {
            Ok(resp) => {
                let config = resp.hotshot_config();
                break (config.blocks_per_epoch(), config.epoch_start_block());
            },
            Err(e) => {
                tracing::error!("Failed to fetch the network config: {e}");
                sleep(Duration::from_secs(5)).await;
            },
        }
    };
    Ok(epoch_config)
}

/// Initialize the stake table from a sequencer node given the epoch number
///
/// Does not error, runs until the stake table is provided.
pub async fn fetch_stake_table_from_sequencer(
    sequencer_url: &Url,
    epoch: Option<<SeqTypes as NodeType>::Epoch>,
    // stake_table_capacity: usize,
) -> Result<Vec<PeerConfig<SeqTypes>>> {
    tracing::info!("Initializing stake table from node for epoch {epoch:?}");

    match epoch {
        Some(epoch) => loop {
            match surf_disco::Client::<tide_disco::error::ServerError, StaticVersion<0, 1>>::new(
                sequencer_url.clone(),
            )
            .get::<Vec<PeerConfig<SeqTypes>>>(&format!("node/stake-table/{}", epoch.u64()))
            .send()
            .await
            {
                Ok(resp) => break Ok(resp),
                Err(e) => {
                    tracing::error!("Failed to fetch the network config: {e}");
                    sleep(Duration::from_secs(5)).await;
                },
            }
        },
        None => loop {
            match surf_disco::Client::<tide_disco::error::ServerError, StaticVersion<0, 1>>::new(
                sequencer_url.clone(),
            )
            .get::<PublicNetworkConfig>("config/hotshot")
            .send()
            .await
            {
                Ok(resp) => break Ok(resp.hotshot_config().known_nodes_with_stake()),
                Err(e) => {
                    tracing::error!("Failed to fetch the network config: {e}");
                    sleep(Duration::from_secs(5)).await;
                },
            }
        },
    }
}

/// Returns both genesis light client state and stake table state
pub async fn light_client_genesis(
    sequencer_url: &Url,
    stake_table_capacity: usize,
) -> anyhow::Result<(LightClientStateSol, StakeTableStateSol)> {
    let st = fetch_stake_table_from_sequencer(sequencer_url, None)
        .await
        .with_context(|| "Failed to initialize stake table")?;
    light_client_genesis_from_stake_table(&st, stake_table_capacity)
}

#[inline]
pub fn light_client_genesis_from_stake_table(
    st: &[PeerConfig<SeqTypes>],
    stake_table_capacity: usize,
) -> anyhow::Result<(LightClientStateSol, StakeTableStateSol)> {
    let st_state = compute_stake_table_commitment(st, stake_table_capacity);
    Ok((
        LightClientStateSol {
            viewNum: 0,
            blockHeight: 0,
            blockCommRoot: U256::from(0u32),
        },
        StakeTableStateSol {
            blsKeyComm: field_to_u256(st_state.bls_key_comm),
            schnorrKeyComm: field_to_u256(st_state.schnorr_key_comm),
            amountComm: field_to_u256(st_state.amount_comm),
            threshold: field_to_u256(st_state.threshold),
        },
    ))
}

use hotshot_stake_table::vec_based::StakeTable;
use hotshot_types::{
    light_client::one_honest_threshold,
    signature_key::BLSPubKey,
    traits::stake_table::{SnapshotVersion, StakeTableScheme},
};

#[inline]
// We'll get rid of it someday
pub fn legacy_light_client_genesis_from_stake_table(
    st: StakeTable<BLSPubKey, StateVerKey, CircuitField>,
) -> anyhow::Result<(LightClientStateSol, StakeTableStateSol)> {
    let (bls_comm, schnorr_comm, stake_comm) = st
        .commitment(SnapshotVersion::LastEpochStart)
        .expect("Commitment computation shouldn't fail.");
    let threshold = one_honest_threshold(st.total_stake(SnapshotVersion::LastEpochStart)?);

    Ok((
        LightClientStateSol {
            viewNum: 0,
            blockHeight: 0,
            blockCommRoot: U256::from(0u32),
        },
        StakeTableStateSol {
            blsKeyComm: field_to_u256(bls_comm),
            schnorrKeyComm: field_to_u256(schnorr_comm),
            amountComm: field_to_u256(stake_comm),
            threshold,
        },
    ))
}

pub fn load_proving_key(stake_table_capacity: usize) -> ProvingKey {
    let srs = {
        let num_gates = crate::circuit::build_for_preprocessing::<
            CircuitField,
            ark_ed_on_bn254::EdwardsConfig,
        >(stake_table_capacity)
        .unwrap()
        .0
        .num_gates();

        tracing::info!("Loading SRS from Aztec's ceremony...");
        let srs_timer = Instant::now();
        let srs = ark_srs::kzg10::aztec20::setup(num_gates + 2).expect("Aztec SRS fail to load");
        let srs_elapsed = Instant::now().signed_duration_since(srs_timer);
        tracing::info!("Done in {srs_elapsed:.3}");

        // convert to Jellyfish type
        // TODO: (alex) use constructor instead https://github.com/EspressoSystems/jellyfish/issues/440
        UnivariateUniversalParams {
            powers_of_g: srs.powers_of_g,
            h: srs.h,
            beta_h: srs.beta_h,
            powers_of_h: vec![srs.h, srs.beta_h],
        }
    };

    tracing::info!("Generating proving key and verification key.");
    let key_gen_timer = Instant::now();
    let (pk, _) = crate::snark::preprocess(&srs, stake_table_capacity)
        .expect("Fail to preprocess state prover circuit");
    let key_gen_elapsed = Instant::now().signed_duration_since(key_gen_timer);
    tracing::info!("Done in {key_gen_elapsed:.3}");
    pk
}

#[inline(always)]
/// Get the latest LightClientState and signature bundle from Sequencer network
pub async fn fetch_latest_state<ApiVer: StaticVersionType>(
    client: &Client<ServerError, ApiVer>,
) -> Result<StateSignaturesBundle, ServerError> {
    tracing::info!("Fetching the latest state signatures bundle from relay server.");
    client
        .get::<StateSignaturesBundle>("/api/state")
        .send()
        .await
}

/// Read the following info from the LightClient contract storage on chain
/// - latest finalized light client state
/// - stake table commitment used in currently active epoch
///
/// Returned types are of Rust struct defined in `hotshot-types`.
pub async fn read_contract_state(
    provider: impl Provider,
    address: Address,
) -> Result<(LightClientState, StakeTableState), ProverError> {
    let contract = LightClientV2::new(address, &provider);
    let state: LightClientStateSol = match contract.finalizedState().call().await {
        Ok(s) => s.into(),
        Err(e) => {
            tracing::error!("unable to read finalized_state from contract: {}", e);
            return Err(ProverError::ContractError(e.into()));
        },
    };
    let st_state: StakeTableStateSol = match contract.votingStakeTableState().call().await {
        Ok(s) => s.into(),
        Err(e) => {
            tracing::error!(
                "unable to read genesis_stake_table_state from contract: {}",
                e
            );
            return Err(ProverError::ContractError(e.into()));
        },
    };

    Ok((state.into(), st_state.into()))
}

/// Submit the latest finalized state along with a proof to a single light client contract.
///
/// This function is an internal implementation detail used by `submit_state_and_proof` to submit
/// proofs to individual contracts. It handles converting the proof and inputs to the format
/// expected by the contract, sending the transaction, and checking the result.
///
/// Returns the transaction receipt on success or an error if the submission fails.
async fn submit_state_and_proof_to_contract(
    provider: &impl Provider,
    address: Address,
    proof: &Proof,
    public_input: &PublicInput,
) -> Result<TransactionReceipt, ProverError> {
    let contract = LightClientV2::new(address, provider);
    // prepare the input the contract call and the tx itself
    let proof_sol: PlonkProofSol = proof.clone().into();
    let new_state: LightClientStateSol = public_input.lc_state.into();
    let next_stake_table: StakeTableStateSol = public_input.next_st_state.into();

    let tx =
        contract.newFinalizedState_1(new_state.into(), next_stake_table.into(), proof_sol.into());
    tracing::debug!(
        "Sending newFinalizedState tx: address={}, new_state={}, next_stake_table={}\n full tx={:?}",
        address,
        public_input.lc_state,
        public_input.next_st_state,
        tx
    );
    // send the tx
    let (receipt, included_block) = sequencer_utils::contract_send(&tx)
        .await
        .map_err(ProverError::ContractError)?;

    tracing::info!(
        "Submitted state and proof to contract: tx=0x{:x} block={included_block}; success={}",
        receipt.transaction_hash,
        receipt.inner.status()
    );
    if !receipt.inner.is_success() {
        return Err(ProverError::ContractError(anyhow!("{:?}", receipt)));
    }

    Ok(receipt)
}

/// Submit the latest finalized state along with a proof to all configured light client contracts.
///
/// This function first submits the proof to the primary contract address, and if successful,
/// it attempts to submit the same proof to any additional contract addresses configured.
/// The additional contract submissions are non-blocking - failures will be logged but won't
/// cause the function to error.
///
/// Returns the transaction receipt from the primary contract submission.
pub async fn submit_state_and_proof(
    provider: impl Provider,
    address: Address,
    proof: Proof,
    public_input: PublicInput,
) -> Result<TransactionReceipt, ProverError> {
    // Submit to primary contract first
    let primary_receipt =
        submit_state_and_proof_to_contract(&provider, address, &proof, &public_input).await?;

    // Get additional addresses from the thread-local context if this was called from run_prover_service
    if let Some(state) = CURRENT_PROVER_STATE
        .try_with(|state| state.borrow().clone())
        .ok()
        .flatten()
    {
        // Submit to additional contracts if any
        for additional_address in &state.config.additional_light_client_addresses {
            match submit_state_and_proof_to_contract(
                &provider,
                *additional_address,
                &proof,
                &public_input,
            )
            .await
            {
                Ok(_) => {
                    tracing::info!(
                        "Successfully submitted proof to additional contract: {:?}",
                        additional_address
                    );
                },
                Err(err) => {
                    // Log error but continue with other contracts
                    tracing::error!(
                        "Failed to submit proof to additional contract {:?}: {:?}",
                        additional_address,
                        err
                    );
                },
            }
        }
    } else {
        tracing::debug!("No additional light client addresses available in the current context");
    }

    // Return the receipt from the primary contract
    Ok(primary_receipt)
}

async fn fetch_epoch_state_from_sequencer(
    sequencer_url: &Url,
    epoch: u64,
) -> Result<LightClientStateUpdateCertificate<SeqTypes>, ProverError> {
    let state_cert =
        surf_disco::Client::<tide_disco::error::ServerError, StaticVersion<0, 1>>::new(
            sequencer_url.clone(),
        )
        .get::<StateCertQueryData<SeqTypes>>(&format!("availability/state-cert/{}", epoch))
        .send()
        .await?;
    Ok(state_cert.0)
}

async fn generate_proof(
    state: &mut ProverServiceState,
    light_client_state: LightClientState,
    current_stake_table_state: StakeTableState,
    next_stake_table_state: StakeTableState,
    signature_map: HashMap<StateVerKey, StateSignature>,
    proving_key: &ProvingKey,
) -> Result<(Proof, PublicInput), ProverError> {
    // Stake table update is already handled in the epoch catchup
    let entries = state
        .stake_table
        .iter()
        .map(|entry| {
            (
                entry.state_ver_key.clone(),
                entry.stake_table_entry.stake_amount,
            )
        })
        .collect::<Vec<_>>();
    let mut signer_bit_vec = vec![false; entries.len()];
    let mut signatures = vec![Default::default(); entries.len()];
    let mut accumulated_weight = U256::ZERO;
    entries.iter().enumerate().for_each(|(i, (key, stake))| {
        if let Some(sig) = signature_map.get(key) {
            // Check if the signature is valid
            if key.verify_state_sig(sig, &light_client_state, &next_stake_table_state) {
                signer_bit_vec[i] = true;
                signatures[i] = sig.clone();
                accumulated_weight += *stake;
            } else {
                tracing::info!("Invalid signature for key: {:?}", key);
            }
        }
    });

    if accumulated_weight < field_to_u256(current_stake_table_state.threshold) {
        return Err(ProverError::InvalidState(
            "The signers' total weight doesn't reach the threshold.".to_string(),
        ));
    }

    tracing::info!("Collected latest state and signatures. Start generating SNARK proof.");
    let proof_gen_start = Instant::now();
    let proving_key_clone = proving_key.clone();
    let stake_table_capacity = state.config.stake_table_capacity;
    let (proof, public_input) = spawn_blocking(move || {
        generate_state_update_proof(
            &mut ark_std::rand::thread_rng(),
            &proving_key_clone,
            entries,
            signer_bit_vec,
            signatures,
            &light_client_state,
            &current_stake_table_state,
            stake_table_capacity,
            &next_stake_table_state,
        )
    })
    .await
    .map_err(|e| ProverError::Internal(format!("failed to join task: {e}")))??;

    let proof_gen_elapsed = Instant::now().signed_duration_since(proof_gen_start);
    tracing::info!("Proof generation completed. Elapsed: {proof_gen_elapsed:.3}");

    Ok((proof, public_input))
}

/// This function will fetch the cross epoch state update information from the sequencer query node
/// and update the light client state in the contract to the `target_epoch`.
/// In the end, both the locally stored stake table and the contract light client state will correspond
/// to the `target_epoch`.
/// It returns the final stake table state at the target epoch.
async fn advance_epoch(
    state: &mut ProverServiceState,
    provider: &impl Provider,
    light_client_address: Address,
    mut cur_st_state: StakeTableState,
    proving_key: &ProvingKey,
    contract_epoch: Option<<SeqTypes as NodeType>::Epoch>,
    target_epoch: Option<<SeqTypes as NodeType>::Epoch>,
) -> Result<StakeTableState, ProverError> {
    let Some(target_epoch) = target_epoch else {
        return Err(ProverError::Internal(
            "Shouldn't be called pre-epoch.".to_string(),
        ));
    };
    // First sync the local stake table if necessary.
    if state.epoch != contract_epoch {
        state
            .sync_with_epoch(contract_epoch)
            .await
            .map_err(ProverError::NetworkError)?;
    }
    let base_epoch = contract_epoch
        .map(|en| en.u64())
        .unwrap_or(0)
        .max(epoch_from_block_number(
            state.config.epoch_start_block,
            state.config.blocks_per_epoch,
        ));
    let target_epoch = target_epoch.u64();
    for epoch in base_epoch..target_epoch {
        tracing::info!("Performing epoch root state update for epoch {epoch}...");
        let state_cert =
            fetch_epoch_state_from_sequencer(&state.config.sequencer_url, epoch).await?;
        let signature_map = state_cert
            .signatures
            .into_iter()
            .collect::<HashMap<StateVerKey, StateSignature>>();

        let (proof, public_input) = generate_proof(
            state,
            state_cert.light_client_state,
            cur_st_state,
            state_cert.next_stake_table_state,
            signature_map,
            proving_key,
        )
        .await?;

        // Submit to primary and additional contracts
        submit_state_and_proof(provider, light_client_address, proof, public_input).await?;

        tracing::info!("Epoch root state update successfully for epoch {epoch}.");

        state
            .sync_with_epoch(Some(EpochNumber::new(epoch + 1)))
            .await
            .map_err(ProverError::NetworkError)?;
        cur_st_state = state_cert.next_stake_table_state;
    }
    Ok(cur_st_state)
}

/// Sync the light client state from the relay server and submit the proof to the L1 LightClient contract
pub async fn sync_state<ApiVer: StaticVersionType>(
    state: &mut ProverServiceState,
    proving_key: &ProvingKey,
    relay_server_client: &Client<ServerError, ApiVer>,
) -> Result<(), ProverError> {
    let light_client_address = state.config.light_client_address;
    let wallet = EthereumWallet::from(state.config.signer.clone());
    let provider = ProviderBuilder::new()
        .wallet(wallet)
        .on_http(state.config.provider_endpoint.clone());

    tracing::info!(
        ?light_client_address,
        "Start syncing light client state for provider: {}",
        state.config.provider_endpoint,
    );

    let blocks_per_epoch = state.config.blocks_per_epoch;
    let epoch_start_block = state.config.epoch_start_block;

    let (contract_state, mut contract_st_state) =
        read_contract_state(&provider, light_client_address).await?;
    tracing::info!(
        "Current HotShot block height on contract: {}",
        contract_state.block_height
    );

    let bundle = fetch_latest_state(relay_server_client).await?;
    tracing::debug!("Bundle accumulated weight: {}", bundle.accumulated_weight);
    tracing::info!("Latest HotShot block height: {}", bundle.state.block_height);

    if contract_state.block_height >= bundle.state.block_height {
        tracing::info!("No update needed.");
        return Ok(());
    }
    tracing::debug!("Old state: {contract_state:?}");
    tracing::debug!("New state: {:?}", bundle.state);

    tracing::debug!("Contract st state: {contract_st_state}");
    tracing::debug!("Bundle st state: {}", bundle.next_stake);

    let contract_state_epoch_enabled = contract_state.block_height >= epoch_start_block;
    let epoch_enabled = bundle.state.block_height >= epoch_start_block;

    if !epoch_enabled {
        // If epoch hasn't been enabled, directly update the contract.
        let (proof, public_input) = generate_proof(
            state,
            bundle.state,
            contract_st_state,
            contract_st_state,
            bundle.signatures,
            proving_key,
        )
        .await?;

        submit_state_and_proof(&provider, light_client_address, proof, public_input).await?;

        tracing::info!("Successfully synced light client state.");
    } else {
        // After the epoch is enabled
        let contract_epoch = option_epoch_from_block_number::<SeqTypes>(
            contract_state_epoch_enabled,
            contract_state.block_height,
            blocks_per_epoch,
        );
        // If the last contract update was on an epoch root, it's already on the next epoch.
        let contract_epoch = if contract_state_epoch_enabled
            && is_epoch_root(contract_state.block_height, blocks_per_epoch)
        {
            contract_epoch.map(|en| en + 1)
        } else {
            contract_epoch
        };

        let bundle_epoch = option_epoch_from_block_number::<SeqTypes>(
            epoch_enabled,
            bundle.state.block_height,
            blocks_per_epoch,
        );
        let bundle_next_epoch = bundle_epoch.map(|en| en + 1);

        // Update the local stake table if necessary
        if contract_epoch != state.epoch {
            state
                .sync_with_epoch(contract_epoch)
                .await
                .map_err(ProverError::NetworkError)?;
        }

        // A catchup is needed if the contract epoch is behind.
        if bundle_epoch > state.epoch {
            tracing::info!(
                "Catching up from epoch {contract_epoch:?} to epoch {bundle_epoch:?}..."
            );
            contract_st_state = advance_epoch(
                state,
                &provider,
                light_client_address,
                contract_st_state,
                proving_key,
                contract_epoch,
                bundle_epoch,
            )
            .await?;
        }

        // Now that the contract epoch should be equal to the bundle epoch.

        if is_ge_epoch_root(bundle.state.block_height as u64, blocks_per_epoch) {
            // If we reached the epoch root, proceed to the next epoch directly
            // In theory this should never happen because the node won't sign them.
            tracing::info!("Epoch reaching an end, proceed to the next epoch...");
            advance_epoch(
                state,
                &provider,
                light_client_address,
                contract_st_state,
                proving_key,
                bundle_epoch,
                bundle_next_epoch,
            )
            .await?;
        } else {
            // Otherwise process the bundle update information as usual
            let (proof, public_input) = generate_proof(
                state,
                bundle.state,
                contract_st_state,
                contract_st_state,
                bundle.signatures,
                proving_key,
            )
            .await?;

            submit_state_and_proof(&provider, light_client_address, proof, public_input).await?;

            tracing::info!("Successfully synced light client state.");
        }
    }
    Ok(())
}

fn start_http_server<ApiVer: StaticVersionType + 'static>(
    port: u16,
    light_client_address: Address,
    additional_light_client_addresses: Vec<Address>,
    bind_version: ApiVer,
) -> io::Result<()> {
    let mut app = tide_disco::App::<_, ServerError>::with_state(());
    let toml = toml::from_str::<toml::value::Value>(include_str!("../api/prover-service.toml"))
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;

    let mut api = Api::<_, ServerError, ApiVer>::new(toml)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;

    // Endpoint for the primary light client contract address
    api.get("getlightclientcontract", move |_, _| {
        async move { Ok(light_client_address) }.boxed()
    })
    .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;

    // Endpoint for all light client contract addresses
    let all_addresses = {
        let mut addresses = vec![light_client_address];
        addresses.extend_from_slice(&additional_light_client_addresses);
        addresses
    };

    api.get("getlightclientcontracts", move |_, _| {
        let addresses = all_addresses.clone();
        async move { Ok(addresses) }.boxed()
    })
    .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;

    app.register_module("api", api)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;

    spawn(app.serve(format!("0.0.0.0:{port}"), bind_version));
    Ok(())
}

pub async fn run_prover_service<ApiVer: StaticVersionType + 'static>(
    config: StateProverConfig,
    bind_version: ApiVer,
) -> Result<()> {
    let mut state = ProverServiceState::new_genesis(config).await?;

    // Store state in thread-local storage for access in submit_state_and_proof
    CURRENT_PROVER_STATE.with(|current_state| {
        *current_state.borrow_mut() = Some(state.clone());
    });

    let stake_table_capacity = state.config.stake_table_capacity;
    tracing::info!("Stake table capacity: {}", stake_table_capacity);

    tracing::info!(
        "Light client address: {:?}",
        state.config.light_client_address
    );

    if !state.config.additional_light_client_addresses.is_empty() {
        tracing::info!(
            "Additional light client addresses: {:?}",
            state.config.additional_light_client_addresses
        );
    }

    let relay_server_client = Arc::new(Client::<ServerError, ApiVer>::new(
        state.config.relay_server.clone(),
    ));

    // Start the HTTP server to get a functioning healthcheck before any heavy computations.
    if let Some(port) = state.config.port {
        if let Err(err) = start_http_server(
            port,
            state.config.light_client_address,
            state.config.additional_light_client_addresses.clone(),
            bind_version,
        ) {
            tracing::error!("Error starting http server: {}", err);
        }
    }

    let proving_key =
        spawn_blocking(move || Arc::new(load_proving_key(state.config.stake_table_capacity)))
            .await?;

    let update_interval = state.config.update_interval;
    let retry_interval = state.config.retry_interval;
    loop {
        // Update the thread-local storage with the latest state before syncing
        CURRENT_PROVER_STATE.with(|current_state| {
            *current_state.borrow_mut() = Some(state.clone());
        });

        if let Err(err) = sync_state(&mut state, &proving_key, &relay_server_client).await {
            tracing::error!("Cannot sync the light client state, will retry: {}", err);
            sleep(retry_interval).await;
        } else {
            tracing::info!("Sleeping for {:?}", update_interval);
            sleep(update_interval).await;
        }
    }
}

/// Run light client state prover once
pub async fn run_prover_once<ApiVer: StaticVersionType>(
    config: StateProverConfig,
    bind_version: ApiVer,
) -> Result<()> {
    let mut state = ProverServiceState::new_genesis(config).await?;

    // Store state in thread-local storage for access in submit_state_and_proof
    CURRENT_PROVER_STATE.with(|current_state| {
        *current_state.borrow_mut() = Some(state.clone());
    });

    let stake_table_capacity = state.config.stake_table_capacity;

    // Set up HTTP server if needed
    if let Some(port) = state.config.port {
        if let Err(err) = start_http_server(
            port,
            state.config.light_client_address,
            state.config.additional_light_client_addresses.clone(),
            bind_version.clone(),
        ) {
            tracing::error!("Error starting http server: {}", err);
        }
    }

    let proving_key =
        spawn_blocking(move || Arc::new(load_proving_key(stake_table_capacity))).await?;
    let relay_server_client = Client::<ServerError, ApiVer>::new(state.config.relay_server.clone());

    for _ in 0..state.config.max_retries {
        // Update the thread-local storage with the latest state before syncing
        CURRENT_PROVER_STATE.with(|current_state| {
            *current_state.borrow_mut() = Some(state.clone());
        });

        match sync_state(&mut state, &proving_key, &relay_server_client).await {
            Ok(_) => return Ok(()),
            Err(err) => {
                tracing::error!("Cannot sync the light client state, will retry: {}", err);
                sleep(state.config.retry_interval).await;
            },
        }
    }
    Err(anyhow::anyhow!("State update failed"))
}

#[derive(Debug, Display)]
pub enum ProverError {
    /// Invalid light client state or signatures: {0}
    InvalidState(String),
    /// Error when communicating with the smart contract: {0}
    ContractError(anyhow::Error),
    /// Error when communicating with the state relay server: {0}
    RelayServerError(ServerError),
    /// Internal error with the stake table: {0}
    StakeTableError(StakeTableError),
    /// Internal error when generating the SNARK proof: {0}
    PlonkError(PlonkError),
    /// Internal error: {0}
    Internal(String),
    /// General network issue: {0}
    NetworkError(anyhow::Error),
}

impl From<ServerError> for ProverError {
    fn from(err: ServerError) -> Self {
        Self::RelayServerError(err)
    }
}

impl From<PlonkError> for ProverError {
    fn from(err: PlonkError) -> Self {
        Self::PlonkError(err)
    }
}

impl From<StakeTableError> for ProverError {
    fn from(err: StakeTableError) -> Self {
        Self::StakeTableError(err)
    }
}

impl std::error::Error for ProverError {}

#[cfg(test)]
mod test {

    use alloy::{node_bindings::Anvil, providers::layers::AnvilProvider, sol_types::SolValue};
    use anyhow::Result;
    use hotshot_contract_adapter::sol_types::LightClientV2Mock;
    use jf_utils::test_rng;
    use sequencer_utils::{
        deployer::{deploy_light_client_proxy, upgrade_light_client_v2, Contracts},
        test_utils::setup_test,
    };

    use super::*;
    use crate::mock_ledger::{
        MockLedger, MockSystemParam, EPOCH_HEIGHT_FOR_TEST, EPOCH_START_BLOCK_FOR_TEST,
        STAKE_TABLE_CAPACITY_FOR_TEST,
    };

    // const MAX_HISTORY_SECONDS: u32 = 864000;
    const NUM_INIT_VALIDATORS: usize = STAKE_TABLE_CAPACITY_FOR_TEST / 2;

    /// This helper function deploy LightClient V1, and its Proxy, then deploy V2 and upgrade the proxy.
    /// Returns the address of the proxy, caller can cast the address to be `LightClientV2` or `LightClientV2Mock`
    async fn deploy_and_upgrade(
        provider: impl Provider,
        contracts: &mut Contracts,
        is_mock_v2: bool,
        genesis_state: LightClientStateSol,
        genesis_stake: StakeTableStateSol,
    ) -> Result<Address> {
        // prepare for V1 deployment
        let admin = provider.get_accounts().await?[0];
        let prover = admin;

        // deploy V1 and proxy (and initialize V1)
        let lc_proxy_addr = deploy_light_client_proxy(
            &provider,
            contracts,
            true, // mock
            genesis_state,
            genesis_stake,
            Some(prover),
            admin,
        )
        .await?;

        // upgrade to V2
        upgrade_light_client_v2(&provider, contracts, is_mock_v2).await?;

        Ok(lc_proxy_addr)
    }

    #[tokio::test]
    async fn test_read_contract_state() -> Result<()> {
        let (anvil, provider) = setup_test().await?;

        // Deploy the contracts
        let mut contracts = Contracts::default();
        let system_params = MockSystemParam {
            blocks_per_epoch: EPOCH_HEIGHT_FOR_TEST as u64,
            epoch_start_block: EPOCH_START_BLOCK_FOR_TEST as u64,
            max_validator_count: NUM_INIT_VALIDATORS,
        };
        let mock_ledger = MockLedger::new(system_params, &mut test_rng());
        let (genesis_lc, genesis_st) = mock_ledger.init_state();
        let genesis_lc: LightClientStateSol = genesis_lc.into();
        let genesis_st: StakeTableStateSol = genesis_st.into();
        let lc_addr = deploy_and_upgrade(
            &provider,
            &mut contracts,
            true, // deploy V2 mock
            genesis_lc,
            genesis_st,
        )
        .await?;

        // Initialize client contract state
        let (lc_state, st_state) = read_contract_state(provider, lc_addr).await?;

        // Test state correctness
        assert_eq!(lc_state.view_num, genesis_lc.viewNum);
        assert_eq!(lc_state.block_height, genesis_lc.blockHeight);
        assert_eq!(lc_state.block_comm_root, U256::from(0u32));

        anvil.shutdown().await?;

        Ok(())
    }

    /// Test `submit_state_and_proof()` function with mock contract
    #[tokio::test]
    async fn test_submit_state_and_proof() -> Result<()> {
        let (anvil, provider) = setup_test().await?;

        // deploy the contracts
        let mut contracts = Contracts::default();
        let system_params = MockSystemParam {
            blocks_per_epoch: EPOCH_HEIGHT_FOR_TEST as u64,
            epoch_start_block: EPOCH_START_BLOCK_FOR_TEST as u64,
            max_validator_count: NUM_INIT_VALIDATORS,
        };
        let mock_ledger = MockLedger::new(system_params, &mut test_rng());
        let (genesis_lc, genesis_st) = mock_ledger.init_state();
        let genesis_lc: LightClientStateSol = genesis_lc.into();
        let genesis_st: StakeTableStateSol = genesis_st.into();
        let lc_addr = deploy_and_upgrade(
            &provider,
            &mut contracts,
            true, // deploy V2 mock
            genesis_lc,
            genesis_st,
        )
        .await?;

        // Setup for additional contract
        let additional_lc_addr = deploy_and_upgrade(
            &provider,
            &mut contracts,
            true, // deploy V2 mock
            genesis_lc,
            genesis_st,
        )
        .await?;

        // Read contract state
        let (lc_state, st_state) = read_contract_state(provider.clone(), lc_addr).await?;

        let (_, _, sigs, proof) = mock_ledger.step_with_proof()?;

        // Create a mock config for the prover
        let config = StateProverConfig {
            relay_server: "http://127.0.0.1:8000".parse().unwrap(),
            update_interval: Duration::from_secs(10),
            retry_interval: Duration::from_secs(2),
            provider_endpoint: "http://127.0.0.1:8545".parse().unwrap(),
            light_client_address: lc_addr,
            additional_light_client_addresses: vec![additional_lc_addr],
            signer: LocalSigner::random(test_rng()),
            sequencer_url: "http://127.0.0.1:8000".parse().unwrap(),
            port: None,
            stake_table_capacity: STAKE_TABLE_CAPACITY_FOR_TEST,
            blocks_per_epoch: EPOCH_HEIGHT_FOR_TEST as u64,
            epoch_start_block: EPOCH_START_BLOCK_FOR_TEST as u64,
            max_retries: 3,
        };

        let mut state = ProverServiceState {
            config,
            epoch: None,
            stake_table: vec![],
            st_state: st_state.clone(),
        };

        // Set the thread-local state for testing
        CURRENT_PROVER_STATE.with(|current_state| {
            *current_state.borrow_mut() = Some(state.clone());
        });

        // Test state and proof submission
        let next_lc_state = LightClientState {
            view_num: lc_state.view_num + 1,
            block_height: lc_state.block_height + 1,
            block_comm_root: lc_state.block_comm_root,
        };

        let public_input = PublicInput {
            lc_state: next_lc_state,
            st_state: st_state.clone(),
            next_st_state: st_state,
        };

        // Submit proof to all configured contracts (primary and additional)
        let receipt =
            submit_state_and_proof(provider.clone(), lc_addr, proof, public_input).await?;
        assert!(receipt.inner.is_success());

        // Verify updated state on the primary contract
        let (updated_lc_state, _) = read_contract_state(provider.clone(), lc_addr).await?;
        assert_eq!(updated_lc_state.view_num, lc_state.view_num + 1);
        assert_eq!(updated_lc_state.block_height, lc_state.block_height + 1);

        // Verify updated state on the additional contract
        let (updated_add_lc_state, _) = read_contract_state(provider, additional_lc_addr).await?;
        assert_eq!(updated_add_lc_state.view_num, lc_state.view_num + 1);
        assert_eq!(updated_add_lc_state.block_height, lc_state.block_height + 1);

        anvil.shutdown().await?;

        Ok(())
    }

    /// Test submitting a single proof to multiple contracts
    #[tokio::test]
    async fn test_submit_to_multiple_contracts() -> Result<()> {
        let (anvil, provider) = setup_test().await?;

        // Deploy three contracts to test with
        let mut contracts = Contracts::default();
        let system_params = MockSystemParam {
            blocks_per_epoch: EPOCH_HEIGHT_FOR_TEST as u64,
            epoch_start_block: EPOCH_START_BLOCK_FOR_TEST as u64,
            max_validator_count: NUM_INIT_VALIDATORS,
        };
        let mock_ledger = MockLedger::new(system_params, &mut test_rng());
        let (genesis_lc, genesis_st) = mock_ledger.init_state();
        let genesis_lc: LightClientStateSol = genesis_lc.into();
        let genesis_st: StakeTableStateSol = genesis_st.into();

        // Deploy primary contract
        let primary_addr = deploy_and_upgrade(
            &provider,
            &mut contracts,
            true, // deploy V2 mock
            genesis_lc,
            genesis_st,
        )
        .await?;

        // Deploy two additional contracts
        let additional_addr1 = deploy_and_upgrade(
            &provider,
            &mut contracts,
            true, // deploy V2 mock
            genesis_lc,
            genesis_st,
        )
        .await?;

        let additional_addr2 = deploy_and_upgrade(
            &provider,
            &mut contracts,
            true, // deploy V2 mock
            genesis_lc,
            genesis_st,
        )
        .await?;

        // Read initial state from all contracts
        let (primary_state, st_state) = read_contract_state(provider.clone(), primary_addr).await?;
        let (additional_state1, _) =
            read_contract_state(provider.clone(), additional_addr1).await?;
        let (additional_state2, _) =
            read_contract_state(provider.clone(), additional_addr2).await?;

        // Ensure all contracts have the same initial state
        assert_eq!(primary_state.view_num, additional_state1.view_num);
        assert_eq!(primary_state.view_num, additional_state2.view_num);
        assert_eq!(primary_state.block_height, additional_state1.block_height);
        assert_eq!(primary_state.block_height, additional_state2.block_height);

        let (_, _, _, proof) = mock_ledger.step_with_proof()?;

        // Create a mock config for the prover with multiple contract addresses
        let config = StateProverConfig {
            relay_server: "http://127.0.0.1:8000".parse().unwrap(),
            update_interval: Duration::from_secs(10),
            retry_interval: Duration::from_secs(2),
            provider_endpoint: "http://127.0.0.1:8545".parse().unwrap(),
            light_client_address: primary_addr,
            additional_light_client_addresses: vec![additional_addr1, additional_addr2],
            signer: LocalSigner::random(test_rng()),
            sequencer_url: "http://127.0.0.1:8000".parse().unwrap(),
            port: None,
            stake_table_capacity: STAKE_TABLE_CAPACITY_FOR_TEST,
            blocks_per_epoch: EPOCH_HEIGHT_FOR_TEST as u64,
            epoch_start_block: EPOCH_START_BLOCK_FOR_TEST as u64,
            max_retries: 3,
        };

        let state = ProverServiceState {
            config,
            epoch: None,
            stake_table: vec![],
            st_state: st_state.clone(),
        };

        // Set the thread-local state for testing
        CURRENT_PROVER_STATE.with(|current_state| {
            *current_state.borrow_mut() = Some(state.clone());
        });

        // Create a new state to update to
        let next_lc_state = LightClientState {
            view_num: primary_state.view_num + 1,
            block_height: primary_state.block_height + 1,
            block_comm_root: primary_state.block_comm_root,
        };

        let public_input = PublicInput {
            lc_state: next_lc_state,
            st_state: st_state.clone(),
            next_st_state: st_state,
        };

        // Submit proof to all configured contracts
        let receipt =
            submit_state_and_proof(provider.clone(), primary_addr, proof, public_input).await?;
        assert!(receipt.inner.is_success());

        // Verify updated state on all contracts
        let (updated_primary_state, _) =
            read_contract_state(provider.clone(), primary_addr).await?;
        let (updated_additional_state1, _) =
            read_contract_state(provider.clone(), additional_addr1).await?;
        let (updated_additional_state2, _) =
            read_contract_state(provider.clone(), additional_addr2).await?;

        // Verify all states updated correctly
        assert_eq!(updated_primary_state.view_num, primary_state.view_num + 1);
        assert_eq!(
            updated_additional_state1.view_num,
            additional_state1.view_num + 1
        );
        assert_eq!(
            updated_additional_state2.view_num,
            additional_state2.view_num + 1
        );

        assert_eq!(
            updated_primary_state.block_height,
            primary_state.block_height + 1
        );
        assert_eq!(
            updated_additional_state1.block_height,
            additional_state1.block_height + 1
        );
        assert_eq!(
            updated_additional_state2.block_height,
            additional_state2.block_height + 1
        );

        // Verify all contracts have the same final state
        assert_eq!(
            updated_primary_state.view_num,
            updated_additional_state1.view_num
        );
        assert_eq!(
            updated_primary_state.view_num,
            updated_additional_state2.view_num
        );
        assert_eq!(
            updated_primary_state.block_height,
            updated_additional_state1.block_height
        );
        assert_eq!(
            updated_primary_state.block_height,
            updated_additional_state2.block_height
        );

        anvil.shutdown().await?;

        Ok(())
    }
}
