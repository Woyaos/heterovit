# Experiment 14: Communication Optimization Ablation

This hardware-free experiment compares activation compression, fused FPGA FFN
residency, one-copy communication paths, and one versus two FPGA input buffers.
It uses the Experiment 12 reference sensitivity profile so every change is an
optimization effect rather than a hardware-parameter change.

The fused task represents `FFN1 + GELU + FFN2` as one FPGA-resident island. Its
GPU fallback retains the sum of the three original task costs. Its FPGA cost
uses the declared Linear speedup, an explicit elementwise speedup for GELU, and
one dispatch overhead. Compression reduces only cross-device bytes and adds
declared non-contending endpoint encode/decode costs.

Run from WSL at the repository root:

```bash
python3 simgrid_baseline/14_optimization_ablation/run_optimization_ablation.py \
  simgrid_baseline/08_coarse_vit_dag/results/vit_batch1_coarse_dag.json \
  simgrid_baseline/12_full_dag_baselines/sensitivity_profile.json \
  simgrid_baseline/14_optimization_ablation/optimization_config.json \
  simgrid_baseline/14_optimization_ablation/results
```

All outputs declare `valid_for_target_prediction=false`. Compression codec
costs, FPGA GELU performance, mapped-memory synchronization, DMA support, and
buffer overlap must be calibrated on the target platform later.

The concise Chinese interpretation is in `RESULTS_CN.md`. Generated numerical
tables are in `results/RESULTS.md`, and `results/validation_report.json`
contains the automated consistency checks.
