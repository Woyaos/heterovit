# SimGrid Step 4: Shared Resource Contention

Run the summary comparison:

```powershell
wsl -d Ubuntu -- python3 /mnt/d/VS_project/heterogeneous/simgrid_baseline/04_shared_resource_contention/compare.py
```

Run the two-request case with its detailed timeline:

```powershell
wsl -d Ubuntu -- python3 /mnt/d/VS_project/heterogeneous/simgrid_baseline/04_shared_resource_contention/contention_sim.py 2 /mnt/d/VS_project/heterogeneous/simgrid_baseline/04_shared_resource_contention/platform.xml
```

Both requests start at simulated time zero and share the same GPU, FPGA, and
PCIe link. This experiment exposes resource contention that a simple sum of
independent per-request latencies does not represent.
