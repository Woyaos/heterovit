# Experiment 13: Design Space and Multi-Request Pipeline

This is the final hardware-free experiment. It combines:

- a full-DAG analytical sweep over three communication paths, four effective
  bandwidths, four fixed latencies, and four FPGA Linear speedups;
- SimGrid replay of representative points with 1, 2, 4, and 8 simultaneous
  inference requests;
- GPU-only, static all-Linear FPGA, and communication-aware placement.

The analytical scheduler was validated against SimGrid in Experiment 12. The
representative concurrency results are replayed directly in SimGrid because
queue and link contention should not be inferred from a serial formula.

Run under WSL from the repository root:

```bash
python3 simgrid_baseline/13_design_space_and_pipeline/run_design_space.py \
  simgrid_baseline/08_coarse_vit_dag/results/vit_batch1_coarse_dag.json \
  simgrid_baseline/12_full_dag_baselines/sensitivity_profile.json \
  simgrid_baseline/13_design_space_and_pipeline/design_space_config.json \
  simgrid_baseline/13_design_space_and_pipeline/results
```

Every profile and output is a sensitivity result and declares
`valid_for_target_prediction=false`.
