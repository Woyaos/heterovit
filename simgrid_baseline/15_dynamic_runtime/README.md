# Experiment 15: Dynamic Dual-Mode Runtime

This experiment replaces the simultaneous-request replay with deterministic
request arrivals from 20 to 120 requests/s. It compares GPU-only, the
latency-oriented communication-aware placement, static all-offload for
throughput, and an adaptive controller. The adaptive controller selects the
latency placement while fewer than four requests are outstanding and switches
new requests to the throughput placement when the backlog reaches four. The
threshold is selected by the included 1/2/4/8 sweep, not assumed in advance.

Compression and decompression are serialized on their source or destination
compute device in this experiment. Results include mean, P50, P95, maximum
latency, 30 ms SLO violation rate, achieved throughput, and policy decisions.

Run after Experiment 14 has generated the fused DAG and full-stack profile:

```bash
python3 simgrid_baseline/15_dynamic_runtime/run_load_sweep.py \
  simgrid_baseline/14_optimization_ablation/results/dags/ffn_fused.json \
  simgrid_baseline/14_optimization_ablation/results/profiles/fusion_compression_one_copy_double_buffer.json \
  simgrid_baseline/15_dynamic_runtime/dynamic_config.json \
  simgrid_baseline/15_dynamic_runtime/results
```

All results are sensitivity experiments and are invalid for target prediction.

Validate that all cases completed and that the expected structural behavior is
present:

```bash
python3 simgrid_baseline/15_dynamic_runtime/validate_results.py \
  simgrid_baseline/15_dynamic_runtime/dynamic_config.json \
  simgrid_baseline/15_dynamic_runtime/results/dynamic_load_sweep.csv \
  simgrid_baseline/15_dynamic_runtime/results_thresholds/threshold_sweep.csv
```
