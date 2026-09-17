# HeteroViT Infra

面向 Vision Transformer 推理的 GPU–FPGA 异构 AI Infra：覆盖画像、自动分块、
通信感知调度、仿真回放、硬件标定与 IRIS 运行时部署桥接。

[![CI](https://github.com/Woyaos/heterovit/actions/workflows/ci.yml/badge.svg)](https://github.com/Woyaos/heterovit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-D22128.svg)](LICENSE)
[![AI Infra](https://img.shields.io/badge/AI%20Infra-GPU%20%2B%20FPGA-6C5CE7.svg)](docs/architecture.md)

[English](README.md) · [架构](docs/architecture.md) ·
[复现说明](docs/reproducibility.md) · [贡献指南](CONTRIBUTING.md) ·
[行为规范](CODE_OF_CONDUCT.md)

> **当前状态：研究原型。** 仓库内 RF880 + Jetson AGX 数值默认是敏感性假设。
> 只有明确标记 `valid_for_target_prediction=true` 的产物才能作为目标硬件预测，
> 现有仿真结果不能写成真实板卡性能。

## 核心价值

本项目不把“可卸载算子数量”当作最终目标，而是显式考虑 PCIe 固定开销、传输量、
排队、设备切换和中间激活。系统从 ViT DAG 中识别合法的完整 FFN 区域，使用动态
规划选择 GPU/FPGA 边界，再在完整 DAG 上回放，最终把计划导出到 IRIS 运行时。

主要创新点：

- 自动识别并选择 `FC1 → GELU → FC2` 连续 FPGA 岛，避免扩大后的 FFN 中间激活
  反复跨设备传输；
- 同时建模计算、通信、队列与低延迟/高吞吐目标的异构调度；
- 将解析估算、SimGrid 回放、标定预测和真实测量分级，防止结论越界；
- 对不完整硬件测量采取 fail-closed 策略，不用乐观默认值偷偷补齐；
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

运行当前整合实验：

```bash
python simgrid_baseline/17_rf880_agx_final/run_final_experiment.py
```

若环境没有 SimGrid，脚本会明确切换到
`execution_mode=analytical_fallback_no_simgrid`，不会把解析结果伪装成事件仿真。

## 推荐阅读路径

1. [系统架构](docs/architecture.md)：理解各模块与数据流；
2. `simgrid_baseline/18_auto_block_partition/README.md`：核心自动分块算法；
3. `simgrid_baseline/16_hardware_calibration/README.md`：上板测量与标定门禁；
4. `iris-main/apps/hetero_vit_runtime/README.md`：IRIS 运行时桥接；
5. [复现说明](docs/reproducibility.md)：结果等级、命令和解释边界。

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

`01`–`15` 保留研究演进过程；对外展示应优先从 `16`–`20` 与 IRIS 桥接开始。

## 开源边界

`iris-main/` 是带独立 BSD-3-Clause 许可证的第三方 IRIS 源码，项目改动集中在
`iris-main/apps/hetero_vit_runtime/`。大型 ONNX 权重、临时硬件手册、个人研究素材和
本机测量输入不会进入 Git。原创部分采用 Apache-2.0，详见 `LICENSE` 和
`THIRD_PARTY_NOTICES.md`。
