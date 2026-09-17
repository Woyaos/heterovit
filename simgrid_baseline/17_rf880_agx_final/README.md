# RF880-AGX Final Experiment

这个目录是 RF880 + Jetson AGX 板卡确定后的最终实验入口。它做的事不是伪造真实硬件结果，而是把当前可执行的建模仿真、策略消融、动态负载和上板测量模板整合成一个闭环。

## 一键运行

```powershell
python simgrid_baseline\17_rf880_agx_final\run_final_experiment.py
```

默认输出在：

```text
simgrid_baseline/17_rf880_agx_final/results/
```

关键产物：

- `final_summary.json`：机器可读的最终摘要。
- `FINAL_RESULTS.md`：论文/周报可读的结果摘要。
- `optimization_ablation/`：通信感知调度、FFN 融合、激活压缩、共享映射、双缓冲、直接 DMA 上界的消融实验。
- `design_space/`：PCIe 带宽、固定延迟、FPGA Linear 加速比的敏感性分析。
- `dynamic_load/`：多请求到达下的 GPU-only、低延迟异构、吞吐优先异构、自适应策略比较。
- `adaptive_threshold/`：自适应调度阈值扫描。
- `measurement_plan/`：真实上板时要填写的 GPU/FPGA task 延迟和 PCIe/DMA 传输延迟模板。

## 当前边界

`rf880_agx_sensitivity_profile.json` 固定了 RF880 + Jetson AGX 的实际拓扑：Jetson 是 PCIe Root Complex，RF880 FPGA 是 PCIe Endpoint，连接宽度是 x4。但里面的 GPU 有效算力、FPGA Linear 加速比、PCIe 有效带宽、DMA 固定延迟仍是敏感性参数，所以所有结果都标记为：

```text
valid_for_target_prediction = false
```

如果当前 Python 环境没有安装 SimGrid，`run_final_experiment.py` 会自动切换到：

```text
execution_mode = analytical_fallback_no_simgrid
```

这个模式使用已有开销模型和通信感知调度器计算结果，不生成 SimGrid 事件时间线。它适合现在没有完整 SimGrid/硬件环境时推进实验闭环。安装好 SimGrid 后，同一个入口会自动运行完整 SimGrid replay。

等硬件可用后，流程变为：

1. 用 `measurement_plan/` 里的 CSV 模板实测 GPU、FPGA、DMA。
2. 用 `simgrid_baseline/16_hardware_calibration/validate_and_build_profile.py` 生成完整 target profile。
3. 将本目录脚本里的 base profile 换成 target profile。
4. 重新运行最终实验，得到可写成真实硬件结果的表格。

## 论文逻辑

这个目录对应论文中的工程实践章节：

1. 从 ViT ONNX 提取任务图。
2. 固定 RF880-AGX 的 GPU/FPGA/PCIe x4 拓扑。
3. 建立计算和通信开销模型。
4. 设计通信感知调度和连续 Linear/FFN 驻留 FPGA。
5. 加入激活压缩、共享映射、双缓冲、直接 DMA 上界等消融。
6. 用动态负载实验说明调度策略在多请求情况下的延迟和吞吐表现。
7. 用上板测量模板把仿真模型迁移为真实硬件模型。
