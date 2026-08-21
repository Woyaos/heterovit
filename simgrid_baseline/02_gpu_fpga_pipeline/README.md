# SimGrid Step 2: GPU-FPGA Pipeline

Run from PowerShell:

```powershell
wsl -d Ubuntu -- python3 /mnt/d/VS_project/heterogeneous/simgrid_baseline/02_gpu_fpga_pipeline/pipeline.py /mnt/d/VS_project/heterogeneous/simgrid_baseline/02_gpu_fpga_pipeline/platform.xml
```

The example contains two virtual compute devices and one PCIe link:

```text
GPU compute -> 4 MB PCIe transfer -> FPGA compute
            -> 4 MB PCIe transfer -> GPU compute
```

The values are teaching parameters, not measurements of the real Jetson or
FPGA. Later they will be replaced by calibrated values.
