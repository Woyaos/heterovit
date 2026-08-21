# GPU-CPU-FPGA 异构 ViT 项目当前状态

## 已经完成

已经从本地 ONNX 得到完整任务图。该模型经结构检查是 ViT-Base/16-like，而不是文件名
暗示的 ViT-Small：12 个 Transformer block、49 个 Linear。原始 ONNX 被整理为可调度
的粗粒度 DAG，并进一步融合为 101 个任务、124 条依赖、37 个 FPGA 候选子图。

SimGrid 中已经实现 GPU-only、全部卸载、IRIS 计算时间式策略、数据局部性策略、通信
感知 EFT 和自适应双模式策略。模型显式包含任务计算、排队、PCIe 固定开销与带宽、
主机中转/共享映射/直接 DMA 三类路径、激活压缩、FFN 融合、FPGA 驻留、双缓冲及
多请求流水线。实验 15 已完成动态到达和 SLO 测试，结果见
`15_dynamic_runtime/RESULTS_CN.md`。

IRIS 3.0 已在 CPU/OpenMP 环境完成源码构建。自定义 `hetero_eft` policy 已接入任务
元数据、GPU/FPGA 计算时间、输入输出字节、PCIe 代价、队列深度和低延迟/高吞吐目标。
包含 101 个任务和 124 条依赖的 IRIS 程序已在两种目标下成功运行。这里验证了运行时
接线和 CPU 回退，没有声称验证真实 GPU/FPGA 性能。

## 当前结论

在参考敏感性点，单独使用压缩或一次拷贝不足以让异构部署获胜；最重要的是把
`FFN1 + GELU + FFN2` 作为连续 FPGA 岛，减少设备往返。融合再叠加 2 倍激活压缩、
一次拷贝和双缓冲后，单请求得到相对 GPU-only 的 1.321 倍条件加速。动态实验中，
异构策略把无 SLO 违约的输入范围从 GPU-only 的低于约 45 req/s 提高到 80 req/s；
120 req/s 时系统仍过载。这些只说明在什么参数区域策略有效，不代表目标硬件结果。

## 为什么现在必须等待硬件

后续唯一缺失的是目标平台实测值和真实 kernel：101 个 GPU 子任务、37 个 FPGA 候选
子图、PCIe/DMA 各方向和数据量、编解码、双向及计算通信重叠。实验 16 已生成 138 行
任务测量表和 60 行传输测量表；校验器在数据缺失时会拒绝生成 target profile。没有
Jetson 与 FPGA，继续调整这些数值只会得到更多假设，不会增加结论可信度。

硬件到位后的逐项操作见
`16_hardware_calibration/HARDWARE_STEPS_CN.md`。测量完成后，同一份 profile 可直接复跑
SimGrid，并重新生成 IRIS manifest；届时主要工作是实现真实 CUDA/FPGA kernel、测量、
校准模拟误差，再与 GPU-only 做最终比较。
