# HeteroViT Infra

面向 Vision Transformer 推理的 GPU–FPGA 异构 AI Infra：覆盖模型图分析、
自动分块、通信感知调度、全图评估、硬件开销画像和 IRIS 运行时集成。

[![CI](https://github.com/Woyaos/heterovit/actions/workflows/ci.yml/badge.svg)](https://github.com/Woyaos/heterovit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-D22128.svg)](LICENSE)
[![AI Infra](https://img.shields.io/badge/AI%20Infra-GPU%20%2B%20FPGA-6C5CE7.svg)](docs/architecture.md)

[English](README.md) · [架构](docs/architecture.md) ·
[复现说明](docs/reproducibility.md) · [贡献指南](CONTRIBUTING.md) ·
[行为规范](CODE_OF_CONDUCT.md)

## 核心价值

本项目不把“可卸载算子数量”当作最终目标，而是显式考虑 PCIe 固定开销、传输量、
排队、设备切换和中间激活。系统从 ViT DAG 中识别合法的完整 FFN 区域，使用动态
规划选择 GPU/FPGA 边界，再在完整 DAG 上回放，最终把计划导出到 IRIS 运行时。

主要创新点：

- 自动识别并选择 `FC1 → GELU → FC2` 连续 FPGA 岛，避免扩大后的 FFN 中间激活
  反复跨设备传输；
- 同时建模计算、通信、队列与低延迟/高吞吐目标的异构调度；
- 在完整 ViT DAG 上比较不同设备放置策略，支持 SimGrid 事件回放与目标开销画像；
- 通过测量计划、profile 校验和端到端评估，将计算与传输开销接入调度决策；
- 将 DP 计划导出为 IRIS manifest，并由自定义 `hetero_eft` policy 执行严格放置；
- 建模单次推理中的 token tile streaming、双缓冲、DMA 提交和计算引擎复用。

## 快速开始

推荐 Python 3.10+。核心测试仅依赖标准库：

```bash
python -m unittest discover -s simgrid_baseline/18_auto_block_partition -p "test*.py" -v
python -m unittest discover -s simgrid_baseline/19_single_inference_tile_streaming -p "test*.py" -v
python -m unittest discover -s simgrid_baseline/20_strong_gpu_baseline -p "test*.py" -v
python -m unittest discover -s simgrid_baseline/16_hardware_calibration -p "test*.py" -v
```

运行整合的分块与调度评估：

```bash
python simgrid_baseline/17_rf880_agx_final/run_final_experiment.py
```

## 推荐阅读路径

1. [系统架构](docs/architecture.md)：理解各模块与数据流；
2. `simgrid_baseline/18_auto_block_partition/README.md`：核心自动分块算法；
3. `simgrid_baseline/16_hardware_calibration/README.md`：硬件开销画像与标定流程；
4. `iris-main/apps/hetero_vit_runtime/README.md`：IRIS 运行时桥接；
5. [复现说明](docs/reproducibility.md)：环境配置与实验命令。

## 目录说明

| 目录 | 作用 |
| --- | --- |
| `simgrid_baseline/18_auto_block_partition/` | 核心自动 FFN 分块与证据门禁 |
| `simgrid_baseline/16_hardware_calibration/` | 测量模板、校验器与目标 profile 构建 |
| `simgrid_baseline/17_rf880_agx_final/` | RF880–AGX 整合实验入口 |
| `simgrid_baseline/19_single_inference_tile_streaming/` | 融合 FFN 的 tile streaming 模型 |
| `simgrid_baseline/20_strong_gpu_baseline/` | TensorRT 强 GPU 基线验收 |
| `simgrid_baseline/12_full_dag_baselines/` | 完整 DAG 策略与公共回放逻辑 |
| `iris-main/apps/hetero_vit_runtime/` | IRIS policy、manifest 与运行时骨架 |

核心入口是 `16`–`20` 模块与 IRIS 运行时集成。

## 开源许可

原创代码采用 [Apache License 2.0](LICENSE)。仓库中的第三方 IRIS 源码保留
BSD-3-Clause 许可证；项目运行时集成位于
`iris-main/apps/hetero_vit_runtime/`，详见
[第三方声明](THIRD_PARTY_NOTICES.md)。
