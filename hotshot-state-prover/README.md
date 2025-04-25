# HotShot State Prover

The HotShot State Prover is a service that generates proofs for light client state updates and submits them to the smart contracts.

## Features

- Generates SNARK proofs for light client state updates
- Submits state updates and proofs to light client contracts
- Supports submitting a single proof to multiple contract instances
- Supports epoch-based updates and state synchronization
- Includes a health check server for monitoring

## Configuration

The state prover is configured through the `StateProverConfig` structure, which includes:

- `relay_server`: URL of the state relay server where sequencers push their signatures
- `update_interval`: Interval between light client state updates
- `retry_interval`: Interval between retries if a state update fails
- `provider_endpoint`: URL of the chain (L1 or L2) JSON-RPC provider
- `light_client_address`: Address of the primary LightClient proxy contract
- `additional_light_client_addresses`: Additional LightClient proxy contract addresses to submit proofs to
- `signer`: Transaction signing key for Ethereum or other L2
- `sequencer_url`: URL of a node providing the HotShot config
- `port`: Optional port for running a basic HTTP server
- `stake_table_capacity`: Stake table capacity for the prover circuit
- `blocks_per_epoch`: Epoch length in number of HotShot blocks
- `epoch_start_block`: The epoch start block
- `max_retries`: Maximum number of retries for one-shot prover

## Running the Prover

The prover can be run in two modes:

1. As a daemon service using `run_prover_service` that continuously checks for updates
2. As a one-shot process using `run_prover_once` that runs once and exits

## Multi-Contract Support

The prover can submit the same proof to multiple contract instances simultaneously. This is useful when the same state needs to be maintained across multiple chains or when multiple contracts on the same chain need to receive the state updates.

To use this feature:
1. Set the primary contract address in `light_client_address`
2. Add additional contract addresses to the `additional_light_client_addresses` vector

The prover will first submit the proof to the primary contract, and only if that succeeds, it will attempt to submit to the additional contracts. Failures in additional contract submissions will be logged but won't cause the overall operation to fail. 
