# GPU-FPGA FFN automatic block partition: reproducible modeling experiment

## Scope and status

This module does not quantize ViT, implement FPGA kernels, modify IRIS, or measure the RF880 board. It uses the existing cost-model APIs in `../12_full_dag_baselines/run_policy.py` and the coarse DAG extracted from `vit.onnx`. The ONNX inspection identifies **ViT-Base/16-like** (86.45M parameters, hidden dimension 768, 12 transformer blocks), not ViT-Small. Hardware topology is Jetson CPU/GPU to RF880 PCIe endpoint, with host-staged transfers as the current conservative assumption. The assumed FPGA whole-FFN/GELU support, memory budget, fourfold Linear speedup and PCIe latency are **unverified**.

## Implemented decision process

1. Identify `fc1 -> GELU -> fc2` chains from actual DAG dependencies. Reject chains with additional producers/consumers or unfixed GPU interfaces. Other ViT residual and attention dependencies remain in the outer DAG.
2. Generate legal singleton GPU/FPGA tasks and declared full FPGA FFN blocks from `capability_assumptions.json`. Full FPGA FFN is legal only while its kernel and workspace are assumed available; the current workspace footprint is a conservative activation estimate, not FPGA synthesis output.
3. For each certified serial FFN region, solve a block/device shortest-path DP. A block compute cost is charged once; a transfer is charged only across device boundaries, including the entry and return edges. The cost path is the same `alpha + bytes / bandwidth` and codec model used by existing baselines.
4. Contract only the selected whole FFN blocks and replay the **entire** quotient DAG with fixed placements and shared GPU, FPGA and link availability. The DP interval cost is not labeled as the full-model latency.
5. Compare GPU-only, capability-declared singleton FFN Linear offloading, split FFN, fixed whole-FFN offloading, and automatic region DP using the same profile and full-DAG replay. The historical all-eligible-Linear FPGA and unrestricted operator-EFT baselines remain counterfactual sensitivity experiments: they include Linear types not declared by the RF880 capability table and are excluded from target-measurement runs.

## Structural evidence added on 2026-09-16

`structural_evidence.py` reads the actual edge sizes of every FFN and independently calculates split-versus-complete boundary bytes and transfer costs. It also compares fixed placements for QKV, projection, each FFN Linear and classifier as **counterfactual singleton-kernel** controls; those extra kernels are not claimed to exist on RF880. Each complete FFN removes two expanded-intermediate crossings: 3,025,920 to 605,184 logical bytes per region at FP16. The inequality and its conditions are recorded in `ISCAS_THEORY_REVISED_BILINGUAL.md`.

The additional one-request SimGrid traces in `results/structural_replay` compare split FFN Linear/GPU GELU (31.129 ms), fixed complete FFN (17.994 ms), and a Linear-only capability that selects GPU-only (24.749 ms). All are sensitivity simulations using assumed kernels and CM02 networking, not target-board measurements. Trace tests check task uniqueness, DAG dependencies, physical transfer-stage counts, and two-stage ordering.

## Commands

From the workspace root:

```powershell
python -m unittest discover -s simgrid_baseline/18_auto_block_partition -p test_partition.py -v
python simgrid_baseline/18_auto_block_partition/partition.py
python simgrid_baseline/18_auto_block_partition/sweep.py
python simgrid_baseline/18_auto_block_partition/nonuniform_example.py
python simgrid_baseline/18_auto_block_partition/structural_evidence.py
```

The first model run writes `results/default/summary.json`, `mixed_dag.json`, `placements.json` and task/transfer timelines. The sweep writes `results/sensitivity.csv` (60 unmeasured parameter combinations). The last command writes a **synthetic** heterogeneous-cost example, not a target benchmark. Re-running does not change the existing IRIS source or the shared scheduler.

SimGrid bindings are available in the local WSL environment. Use one scenario per process because this installed SimGrid version forbids creating two engines in one process. The matrix scripts run and validate all scenarios separately:

```bash
python3 simgrid_baseline/18_auto_block_partition/run_replay_matrix.py
python3 simgrid_baseline/18_auto_block_partition/run_online_matrix.py
```

These scripts must run from WSL/Linux with Python SimGrid bindings. They write 12 batch replay and 9 online-arrival scenarios to `results/replay` and `results/online`. Both use RF880 **sensitivity assumptions**, not measured board parameters. The installed SimGrid 3.35 uses CM02; its Raw PCI-like network model is unavailable in this local version, so the PCIe behavior still requires hardware calibration. A one-request experiment cannot substantiate a double-buffer throughput claim.

The selected quotient DAG is also exported to IRIS as a planned 9-field task manifest. `results/iris_planned_manifest.h` and `.json` match all 101 selected tasks and 124 edges; 12 tasks are planned for FPGA. The default IRIS CPU-only skeleton still runs the automatic (unplanned) manifest, whereas the planned skeleton only compiles until NVIDIA/FPGA devices and real kernels are present.

## Board-stage inputs

Replace the sensitivity profile with the existing complete target-measurement profile format after measuring: GPU task latencies (including launches and conversions), FPGA singleton Linear and whole-FFN kernel latencies (including dispatch), each GPU-host and host-FPGA PCIe direction at several payload sizes, effective transfer bandwidth and fixed latency, FPGA weights/activation memory capacity, buffer count and supported operator/block implementations. Confirm PCIe generation/width on the powered board and whether direct GPU-FPGA DMA is actually supported before selecting a direct path. Rebuild the DAG/profile if the deployed ONNX model or precision differs.

Only a measured, complete profile should set `valid_for_target_prediction=true`. For the partition experiment, the target profile must cover the original and selected fused DAGs together, and the capability status must be `verified_target_capability`. Missing measured task costs now fail rather than falling back to speedup assumptions. Target SimGrid summaries are separately labeled as target-profile event replay and include model/precision metadata for held-out whole-model error analysis. The FP16 union measurement plan and commands are in `../16_hardware_calibration/README.md`. Finally compare simulated vs end-to-end TensorRT GPU-only and actual heterogeneous runtime latency over warm-up and repeated requests, report median/p95 and error, and revise the model if discrepancy is material.

See `EXPERIMENT_REPORT_CN.md` for the recorded results and limitations.
