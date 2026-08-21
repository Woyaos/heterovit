# Experiment 09: Cost Data Interface

Generate blank target-measurement templates:

```powershell
conda run -n hetero_vit python simgrid_baseline/09_cost_data_interface/generate_cost_templates.py simgrid_baseline/08_coarse_vit_dag/results/vit_batch1_coarse_dag.json simgrid_baseline/09_cost_data_interface/results/target_measurement
```

Report completeness without failing:

```powershell
conda run -n hetero_vit python simgrid_baseline/09_cost_data_interface/validate_cost_data.py simgrid_baseline/09_cost_data_interface/results/target_measurement/task_costs.csv simgrid_baseline/09_cost_data_interface/results/target_measurement/platform_profile.json
```

Add `--strict` when launching a formal simulation. Strict validation exits with
code 2 while any required device time or platform field is missing.

The target-measurement templates must never contain invented values. Synthetic
design-space inputs belong in a separate sensitivity profile and must use
`source_kind=sensitivity`.
