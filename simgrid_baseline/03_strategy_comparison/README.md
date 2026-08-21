# SimGrid Step 3: Compare Two Placement Strategies

Run the complete comparison from PowerShell:

```powershell
wsl -d Ubuntu -- python3 /mnt/d/VS_project/heterogeneous/simgrid_baseline/03_strategy_comparison/compare.py
```

Run one case with its detailed timeline:

```powershell
wsl -d Ubuntu -- python3 /mnt/d/VS_project/heterogeneous/simgrid_baseline/03_strategy_comparison/strategy_sim.py offload /mnt/d/VS_project/heterogeneous/simgrid_baseline/03_strategy_comparison/platform_slow_pcie.xml
```

The modeled Linear takes 2 ms on the GPU and 0.4 ms on the FPGA. Offloading
also transfers a 4 MB input and a 4 MB output. The comparison demonstrates
that a faster FPGA kernel is not sufficient: the complete path must include
PCIe communication.
