# Experiment 10: Break-Even Sensitivity Sweep

Run from PowerShell:

```powershell
conda run -n hetero_vit python simgrid_baseline/10_break_even_sensitivity/run_break_even_sweep.py simgrid_baseline/08_coarse_vit_dag/results/vit_batch1_coarse_dag.json simgrid_baseline/10_break_even_sensitivity/sensitivity_config.json simgrid_baseline/10_break_even_sensitivity/results
```

For an FPGA speedup `S = T_gpu / T_fpga`, isolated offload is beneficial when:

```text
T_gpu / S + T_comm < T_gpu
```

The reported break-even GPU latency is therefore:

```text
T_gpu_min = T_comm / (1 - 1 / S)
```

This is a conditional sensitivity study, not a target-hardware prediction. It
assumes preloaded FPGA weights, two boundary transfers, no overlap, and no
queueing. The target-measurement templates in Experiment 09 remain untouched.
