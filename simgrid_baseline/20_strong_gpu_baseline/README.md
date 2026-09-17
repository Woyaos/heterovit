# Experiment 20: Strong GPU Baseline

This experiment prevents a heterogeneous result from being compared against an
unoptimized analytical GPU baseline. On the Jetson target, run:

```bash
bash simgrid_baseline/20_strong_gpu_baseline/run_trtexec_matrix_jetson.sh vit.onnx
```

The script builds FP16 batch-1 TensorRT engines with 0/1/2/4 auxiliary streams
and measures each engine with and without CUDA Graphs and host data transfers.
Record median, p95, mean, throughput, enqueue time, hashes, target identity, and
raw logs in `gpu_baseline_measurements.csv`, then validate:

```bash
python simgrid_baseline/20_strong_gpu_baseline/validate_gpu_baseline.py
```

An incomplete matrix remains `waiting_for_jetson_measurements` and cannot be
used as the paper's GPU baseline. Run `trtexec --help` on the target first and
adapt flags only when the installed TensorRT version requires it.
