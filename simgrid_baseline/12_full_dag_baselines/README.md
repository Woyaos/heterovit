# Experiment 12: Full ViT DAG Baselines

This experiment replays the 125-task ViT DAG in SimGrid and compares five
placement policies. Cross-device dependencies use two explicit transfers:
device-to-host followed by host-to-device.

The included profile is a scheduler-validation sensitivity profile. It is not
a Jetson, FPGA, or PCIe measurement and every output declares
`valid_for_target_prediction=false`.

Run under WSL from the repository root:

```bash
python3 simgrid_baseline/12_full_dag_baselines/compare_policies.py \
  simgrid_baseline/08_coarse_vit_dag/results/vit_batch1_coarse_dag.json \
  simgrid_baseline/12_full_dag_baselines/sensitivity_profile.json \
  simgrid_baseline/12_full_dag_baselines/results
```

Outputs include a policy comparison plus per-policy placement, task timeline,
transfer timeline, generated SimGrid platform, and summary files.

The favorable profile is a second conditional point used to verify that the
communication-aware policy does choose FPGA when end-to-end offload is cheaper:

```bash
python3 simgrid_baseline/12_full_dag_baselines/compare_policies.py \
  simgrid_baseline/08_coarse_vit_dag/results/vit_batch1_coarse_dag.json \
  simgrid_baseline/12_full_dag_baselines/favorable_sensitivity_profile.json \
  simgrid_baseline/12_full_dag_baselines/results_favorable
```
