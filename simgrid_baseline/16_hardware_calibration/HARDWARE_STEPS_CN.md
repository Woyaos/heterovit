# 真实硬件到位后的操作清单

当前代码、DAG、模拟器、IRIS 调度策略和数据接口都已准备好。下面这些步骤必须在
Jetson 与 FPGA 通过 PCIe 连接后完成，不能靠填写理论参数替代。

## 1. 固定实验环境

记录 Jetson 型号、功耗模式、CPU/GPU 锁频方式、CUDA/TensorRT 版本；记录 FPGA
型号、PCIe 代际和通道数、DMA 驱动、bitstream 版本及工作频率。每轮测量前预热，
正式测量至少 30 次，并在每次计时前后做设备同步。

## 2. 测 GPU 和 FPGA 子任务

按照 `results/target_measurement/task_measurements.csv` 的 138 行逐项测量。其中包含
101 个 GPU 子任务和 37 个可放到 FPGA 的 Linear 或融合 FFN 子图。填写均值、P50、
P95、样本数、精度、kernel/bitstream 名称和原始日志路径。这里测的是实际执行时间，
不能使用 FLOPs 推算值。

## 3. 测通信路径

先分别验证三种路径是否可用：普通主机中转、Jetson 共享/映射内存加一次 FPGA DMA、
设备间直接 DMA。通常优先验证共享映射的一次拷贝路径，不能默认 Jetson GPU 与 FPGA
支持 PCIe P2P。按照 `transfer_measurements.csv` 测不同数据量和两个方向，计时必须包含
DMA 提交与完成同步。若选择主机中转，还要分别测 GPU-host 与 host-FPGA 两段。

## 4. 测重叠和压缩

用两个 buffer 验证“第 N 个请求在 FPGA 计算时，第 N+1 个请求能否同时 DMA”；再分别
验证双向 DMA、GPU 计算与 DMA 是否真正并行。激活压缩若启用，单独测编码、解码固定
开销和带宽；若不启用，比例填 1，编解码开销填 0，并保留对应测量日志。

## 5. 生成真实 profile 并复跑

填写 `platform_profile.json` 后执行 `validate_and_build_profile.py`。只有 138 个任务设备
记录、所选通信路径和平台信息全部完整，脚本才会生成
`calibrated_target_profile.json`。然后依次复跑 GPU-only、通信感知 EFT、全卸载和自适应
双模式，报告单请求 P50/P95、吞吐率、30 ms SLO 违约率、传输字节数和设备利用率。

## 6. 接入 IRIS 实机运行

用真实 profile 重新运行 `iris-main/apps/hetero_vit_runtime/generate_manifest.py`，再为
任务名提供 CUDA 与 FPGA kernel。先比较 IRIS 实测时间和 SimGrid 预测误差，误差较大的
任务回到测量表校准；最后才讨论相对 GPU-only 的加速和论文创新点。
