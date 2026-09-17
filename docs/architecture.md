# Architecture

## Design goal

HeteroViT Infra answers one systems question: **when does a heterogeneous
GPU–FPGA placement improve end-to-end ViT inference after communication and
runtime effects are included?** It deliberately separates portable scheduling
logic from target-specific kernels and measurements.

## Control and data path

1. **Graph extraction** converts the model into a coarse task DAG with tensor
   sizes and dependencies.
2. **Capability declaration** states which singleton operators and fused blocks
   a target can execute, including workspace constraints. An assumption is not
   treated as a verified capability.
3. **Cost profile** supplies per-device compute costs and directional transfer
   costs using `latency + bytes / bandwidth`, optionally with codec overhead.
4. **Region certification** recognizes branch-free FFN chains and rejects unsafe
   fusion when there are extra producers, consumers, or unsupported kernels.
5. **Partitioning** solves each legal region using a shortest-path dynamic
   program. A block cost is charged once and a transfer only at a device
   boundary.
6. **Whole-graph replay** contracts selected blocks, preserves external edges,
   and evaluates all tasks with shared GPU, FPGA, and link availability.
7. **Runtime export** writes a nine-field IRIS task manifest. The custom policy
   evaluates calibrated cost, queue depth, communication, and the active
   latency/throughput objective.
8. **Evidence gates** verify artifact consistency and prevent sensitivity data
   from being reported as target measurements.

## Core modules

### Automatic FFN partitioner

`simgrid_baseline/18_auto_block_partition/partition.py` is the primary research
implementation. It keeps the optimizer local to certified serial regions but
evaluates the selected plan on the complete quotient DAG. Exhaustive-reference
tests check the DP, and structural tests check graph contraction and boundaries.

### Scheduling and simulation

`simgrid_baseline/12_full_dag_baselines/run_policy.py` contains shared cost and
policy machinery. Experiments cover GPU-only, static offload, locality-aware and
communication-aware EFT, and adaptive objectives. SimGrid is an optional event
replay backend; a missing binding must remain visible in artifact metadata.

### Calibration

`simgrid_baseline/16_hardware_calibration/` generates the union of measurements
needed by the original and fused DAGs. The profile builder rejects missing,
duplicate, precision-mismatched, or capability-inconsistent data. Held-out
end-to-end latency validation reports median, p95, and prediction error.

### IRIS integration

`iris-main/apps/hetero_vit_runtime/` contains the project-specific runtime work:
manifest generation, the `hetero_eft` custom policy, a full DAG skeleton, and a
CPU-only wiring smoke test. Real CUDA/FPGA kernels and memory objects are target
work; the no-op OpenMP kernel is not a performance implementation.

### Tile streaming

`simgrid_baseline/19_single_inference_tile_streaming/` explores a different
optimization level: overlapping DMA, FC1, GELU, and FC2 across token tiles
inside one inference. It explicitly distinguishes shared and independent
compute engines and charges per-descriptor DMA setup when configured.

## Trust boundaries

- A topology description does not prove effective bandwidth or kernel support.
- A sensitivity profile can demonstrate conditions and trends, not target
  performance.
- SimGrid replays validate modeled scheduling behavior, not silicon timing.
- Only a complete measured profile that passes held-out validation may enable
  `valid_for_target_prediction=true`.
- A hardware result additionally requires repeated end-to-end runs against a
  competitive GPU baseline under the same precision and workload.

## Target implementation still required

- CUDA/TensorRT kernels and profiling on Jetson AGX;
- RF880 singleton/fused kernels with synthesis or co-simulation evidence;
- directional PCIe/DMA measurement, including fixed costs and overlap behavior;
- real IRIS memory objects, kernel dispatch, and failure handling;
- held-out end-to-end validation and comparison with the strong GPU baseline.
