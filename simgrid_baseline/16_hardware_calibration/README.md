# Experiment 16: Hardware Calibration Handoff

This directory is the stopping point before target GPU/FPGA hardware is
required. It converts the fused 101-task ViT DAG into a measurement plan and,
only after every required value is measured, builds a target-valid cost profile
that both SimGrid and the IRIS manifest generator can consume.

Generate a fresh blank plan:

```powershell
conda run -n hetero_vit python simgrid_baseline/16_hardware_calibration/generate_measurement_plan.py `
  simgrid_baseline/14_optimization_ablation/results/dags/ffn_fused.json `
  simgrid_baseline/16_hardware_calibration/results/target_measurement
```

Do not rerun that command after entering hardware measurements because it
intentionally regenerates blank CSV files. Fill `task_measurements.csv`,
`transfer_measurements.csv`, and `platform_profile.json` from synchronized
target measurements. At least 30 measured samples are required per row.

Validate and build the calibrated profile:

```powershell
conda run -n hetero_vit python simgrid_baseline/16_hardware_calibration/validate_and_build_profile.py `
  simgrid_baseline/14_optimization_ablation/results/dags/ffn_fused.json `
  simgrid_baseline/16_hardware_calibration/results/target_measurement/task_measurements.csv `
  simgrid_baseline/16_hardware_calibration/results/target_measurement/transfer_measurements.csv `
  simgrid_baseline/16_hardware_calibration/results/target_measurement/platform_profile.json `
  simgrid_baseline/16_hardware_calibration/results/calibrated_target_profile.json
```

The validator deliberately exits with code 2 while data is missing. It fits
fixed latency and effective bandwidth separately by direction, then uses a
conservative symmetric link in the current simulator. For a host-staged path it
fits GPU-host and host-FPGA links separately.

## RF880 + Jetson FFN partition handoff

The earlier 101-task fused-only plan is insufficient for the latest partition
comparison, which also replays the original 125-task DAG. Generate a fresh
union plan in a **new empty output directory**; do not regenerate a plan after
entering measurements. The FP16 plan includes the actual 302,592-byte FFN
entry/exit and 1,210,368-byte expanded activation payloads:

```powershell
python simgrid_baseline/16_hardware_calibration/generate_measurement_plan.py `
  simgrid_baseline/08_coarse_vit_dag/results/vit_batch1_coarse_dag.json `
  simgrid_baseline/16_hardware_calibration/results/rf880_fp16_union_measurement_plan `
  --additional-dag simgrid_baseline/18_auto_block_partition/results/default/mixed_dag.json `
  --capability simgrid_baseline/18_auto_block_partition/capability_assumptions.json `
  --activation-bytes-per-element 2
```

This blank plan has 137 unique tasks and 173 task/device rows (137 GPU, 36
FPGA), plus `heldout_end_to_end_measurements.csv` for separate whole-model
validation. Only supported singleton FFN Linear and declared full-FFN blocks
require FPGA measurements. The generator refuses to overwrite CSV or platform
files once measurements are entered. The current capability file is an **unverified design
assumption**; on the target, record the actual supported kernels, precision,
workspace and reserved weights in a separate capability file and set its status
to `verified_target_capability` only after verification. If the board has no
FPGA GELU/full-FFN kernel, first rerun `partition.py` with the Linear-only
capability into a new output directory. Use that no-fusion `mixed_dag.json` as
the additional DAG, then generate its measurement plan in another new
directory; the current full-FFN sensitivity speedup cannot be reused as a
target prediction.

When measured rows and platform data are complete, validate the same union:

```powershell
python simgrid_baseline/16_hardware_calibration/validate_and_build_profile.py `
  simgrid_baseline/08_coarse_vit_dag/results/vit_batch1_coarse_dag.json `
  simgrid_baseline/16_hardware_calibration/results/rf880_fp16_union_measurement_plan/task_measurements.csv `
  simgrid_baseline/16_hardware_calibration/results/rf880_fp16_union_measurement_plan/transfer_measurements.csv `
  simgrid_baseline/16_hardware_calibration/results/rf880_fp16_union_measurement_plan/platform_profile.json `
  simgrid_baseline/16_hardware_calibration/results/rf880_fp16_union_measurement_plan/calibrated_profile.json `
  --additional-dag simgrid_baseline/18_auto_block_partition/results/default/mixed_dag.json `
  --capability simgrid_baseline/18_auto_block_partition/capability_verified_target.json
```

The last capability path is a target-stage file to create from verified kernel,
precision, workspace and weight-reservation facts; it does not exist yet. The
validator checks task precision against platform precision and refuses
unverified capability files. The cost model refuses missing measured task costs
in any target profile.

After calibrated target SimGrid replays, collect at least 30 **separate**
whole-model latency samples per policy in
`heldout_end_to_end_measurements.csv`. Use `source_kind=measured`,
`dataset_role=held_out_validation`, unique sample IDs, matching model and
activation precision, and a real provenance reference. Do not reuse the task
or transfer fit samples. Run `validate_heldout_latency.py` with the CSV, output
JSON and repeated `--prediction POLICY=path/to/simgrid_replay_summary.json`
arguments for GPU-only, split FFN, fixed FFN and automatic mapping. It reports
hardware median/p95 and predicted-vs-hardware median percentage error; it
refuses sensitivity replay summaries. Only then can target prediction accuracy
be assessed.

After a complete profile is built, rerun Experiment 12 policies with that
profile and regenerate `vit_task_manifest.h` for IRIS. Results may only be
called target predictions when the generated profile says
`valid_for_target_prediction=true`; sensitivity profiles remain false.
