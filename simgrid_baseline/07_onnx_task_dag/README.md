# Experiment 07: Raw ONNX Task DAG

Run from PowerShell:

```powershell
conda run -n hetero_vit python simgrid_baseline/07_onnx_task_dag/build_task_dag.py vit.onnx simgrid_baseline/07_onnx_task_dag/results
```

The experiment fixes the dynamic input batch to one in memory, runs ONNX shape
inference, and exports operator dependencies. Linear MatMul/Gemm nodes with a
constant 2D weight are marked as GPU/FPGA candidates. Other runtime operators
are GPU-only in the initial research scope.

The output is deliberately a raw ONNX DAG. It will be coarsened into scheduling
tasks in a later experiment because shape and constant-management operators are
too fine-grained for practical device scheduling.
