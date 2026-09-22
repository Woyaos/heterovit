# HeteroViT Infra

**Communication-aware GPU–FPGA infrastructure for ViT graph analysis,
automatic partitioning, scheduling evaluation, and runtime integration.**

[![CI](https://github.com/Woyaos/heterovit/actions/workflows/ci.yml/badge.svg)](https://github.com/Woyaos/heterovit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-D22128.svg)](LICENSE)
[![AI Infra](https://img.shields.io/badge/AI%20Infra-GPU%20%2B%20FPGA-6C5CE7.svg)](docs/architecture.md)

[中文说明](README.zh-CN.md) · [Architecture](docs/architecture.md) ·
[Reproducibility](docs/reproducibility.md) · [Contributing](CONTRIBUTING.md) ·
[Code of Conduct](CODE_OF_CONDUCT.md)

## Why this project

Offloading every eligible operator to an accelerator can lose to GPU-only
execution once PCIe setup, transfers, queueing, and intermediate activations are
included. HeteroViT Infra treats placement as an end-to-end systems problem:
it extracts a schedulable ViT DAG, recognizes legal FFN regions, chooses device
boundaries with a communication-aware cost model, replays the complete graph,
and exports the selected plan to an IRIS runtime integration.

## Highlights

- **Automatic block partitioning.** A shortest-path dynamic program selects
  GPU tasks, FPGA tasks, or complete `FC1 → GELU → FC2` FPGA islands while
  preserving the original DAG dependencies.
- **Communication-aware scheduling.** Compute, queue availability, fixed DMA
  cost, bandwidth, device transitions, activation compression, and request
  objectives participate in placement decisions.
- **End-to-end evaluation.** The workflow compares placement policies on the
  complete ViT DAG and supports SimGrid event replay and calibrated profiles.
- **Hardware profiling pipeline.** Measurement plans and profile validation
  connect per-task and transfer costs to scheduling decisions.
- **Runtime bridge.** A custom IRIS `hetero_eft` policy consumes the exported
  task manifest and supports strict DP-planned placement.
- **Single-inference streaming model.** Token tiling explores buffering, DMA
  submission cost, shared versus independent compute engines, and on-chip
  memory trade-offs inside a fused FFN.

## System flow

```text
ViT ONNX / coarse DAG
        │
        ▼
capability checks + compute/transfer cost profile
        │
        ▼
automatic FFN partition ──► full-DAG policy evaluation
        │                                  │
        ▼                                  ▼
IRIS task manifest              SimGrid replay + profiling
        │
        ▼
hetero_eft runtime policy
```

See [docs/architecture.md](docs/architecture.md) for the architecture and
module interfaces.

## Quick start

Python 3.10+ is recommended. The core validation suite uses only the standard
library; PyTorch, ONNX, SimGrid, CUDA, and IRIS are optional and needed only for
their corresponding stages.

```bash
python -m unittest discover -s simgrid_baseline/18_auto_block_partition -p "test*.py" -v
python -m unittest discover -s simgrid_baseline/19_single_inference_tile_streaming -p "test*.py" -v
python -m unittest discover -s simgrid_baseline/20_strong_gpu_baseline -p "test*.py" -v
python -m unittest discover -s simgrid_baseline/16_hardware_calibration -p "test*.py" -v
```

Run the integrated partitioning and evaluation workflow:

```bash
python simgrid_baseline/17_rf880_agx_final/run_final_experiment.py
```

For environment setup and experiment options, see
[docs/reproducibility.md](docs/reproducibility.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `simgrid_baseline/18_auto_block_partition/` | Main automatic FFN partitioner and evidence gates |
| `simgrid_baseline/16_hardware_calibration/` | Measurement plans, validation, target-profile builder |
| `simgrid_baseline/17_rf880_agx_final/` | Integrated RF880–AGX experiment driver |
| `simgrid_baseline/19_single_inference_tile_streaming/` | Fused-FFN tile-streaming model |
| `simgrid_baseline/20_strong_gpu_baseline/` | TensorRT GPU baseline acceptance gate |
| `simgrid_baseline/12_full_dag_baselines/` | Full-DAG policies and shared cost/replay machinery |
| `iris-main/apps/hetero_vit_runtime/` | Project-specific IRIS policy and manifest bridge |
| `analysis/`, `runtime/`, `models/` | Early cost-model and scheduling prototypes |

Directories `01`–`15` preserve the incremental experimental lineage. The
recommended public entry points are `16`–`20` and the IRIS integration above.

## License and third-party software

Original HeteroViT Infra code is released under the [Apache License 2.0](LICENSE).
The vendored IRIS source retains its BSD-3-Clause license; project-specific
runtime integration is in `iris-main/apps/hetero_vit_runtime/`. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
