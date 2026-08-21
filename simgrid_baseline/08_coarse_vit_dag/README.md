# Experiment 08: Coarse ViT Scheduling DAG

Run from PowerShell:

```powershell
conda run -n hetero_vit python simgrid_baseline/08_coarse_vit_dag/build_coarse_dag.py simgrid_baseline/07_onnx_task_dag/results/vit_batch1_raw_dag.json simgrid_baseline/08_coarse_vit_dag/results
```

Each Transformer block is represented by ten scheduling tasks: two layer
normalizations, four Linear operations, self-attention, GELU, and two residual
adds. Residual dependency edges are preserved. Linear tasks can run on GPU or
FPGA; all other tasks are initially GPU-only.

The graph stores real batch-1 FP32 tensor byte counts but contains no GPU or
FPGA execution-time assumptions.
