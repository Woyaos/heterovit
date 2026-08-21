# SimGrid Step 1: One Virtual Device

Run from PowerShell:

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/d/VS_project/heterogeneous/simgrid_baseline/01_hello && python3 hello_simgrid.py platform.xml"
```

The virtual GPU has a speed of `1 Gf/s`. The task contains `10,000,000`
simulated operations, so its expected simulated duration is:

```text
10,000,000 / 1,000,000,000 = 0.01 seconds
```

This is simulated time. The script should complete almost immediately in real
time while reporting `simulated_time_s=0.010000`.
