# Experiment 06: Linear Activation Tensor Sizes

Run from PowerShell:

```powershell
conda run -n hetero_vit python simgrid_baseline/06_linear_tensor_sizes/extract_linear_tensors.py vit.onnx simgrid_baseline/06_linear_tensor_sizes/results/linear_tensor_sizes.csv
```

The CSV reports batch-1 FP32 activation bytes for every offloadable Linear.
`round_trip_bytes` is the input sent to the FPGA plus the output returned to
the GPU. It intentionally excludes model weights under the assumption that
weights are loaded into FPGA-attached memory before inference.
