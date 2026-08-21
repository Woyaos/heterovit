# Experiment 05: Identify the ONNX Workload

Run from PowerShell with the existing `hetero_vit` environment:

```powershell
conda run -n hetero_vit python simgrid_baseline/05_vit_workload/inspect_vit_onnx.py vit.onnx simgrid_baseline/05_vit_workload/results
```

Outputs:

- `results/model_summary.txt`: model identity, dimensions, parameter count, and operation counts.
- `results/linear_layers.csv`: every ONNX MatMul/Gemm node with a constant 2D weight.

This experiment reads the ONNX graph only. It does not benchmark target-device
latency and does not modify the original model.
