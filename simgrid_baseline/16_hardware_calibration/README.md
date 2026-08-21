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

After a complete profile is built, rerun Experiment 12 policies with that
profile and regenerate `vit_task_manifest.h` for IRIS. Results may only be
called target predictions when the generated profile says
`valid_for_target_prediction=true`; sensitivity profiles remain false.
