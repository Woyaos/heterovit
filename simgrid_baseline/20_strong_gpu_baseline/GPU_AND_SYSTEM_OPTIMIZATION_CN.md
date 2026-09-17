# GPU与异构系统全链路优化准则

## 当前判断

现有24.749 ms GPU-only来自敏感性公式，不是强GPU基线。最终异构结果必须超过同一ONNX、
同一batch、同一精度下经过TensorRT优化的最佳Jetson GPU结果，否则不能声称系统加速。

## GPU侧

1. TensorRT负责常规层融合、tactic选择和Tensor Core映射。不能用逐ONNX节点时间相加代替。
2. 固定batch 1和输入尺寸，分别构建0、1、2、4条辅助流的FP16 engine。
3. 每个engine分别测试普通enqueue和CUDA Graph。Graph主要降低kernel提交开销。
4. 辅助流只在存在独立分支且GPU仍有空闲资源时有效，也可能因同步和显存增加而变慢。
5. 分别报告GPU计算时间和包含输入输出的数据路径时间，不能混为一个基线。
6. 使用Nsight Systems和TensorRT逐层profile确认融合、并行及enqueue瓶颈。

单个ViT请求的大部分算子具有严格前后依赖，因此人为把相邻依赖算子放到多个CUDA stream
通常不能并行。GPU单请求优化的优先级应为TensorRT融合、CUDA Graph、内存路径，然后才
是由TensorRT判断是否启用辅助流。GPU内部GEMM本身已经采用分块并行，不应在没有profile
证据时自行重写矩阵乘法。

## FPGA侧

- 完整FFN驻留，消除4倍扩张激活的GPU-FPGA往返。
- 比较共享矩阵引擎与独立FC1/FC2引擎；后者速度更高但资源更多。
- 对单请求使用token tile数据流；优先验证整张量DMA加FPGA内部流水。
- 只有驱动支持单次描述符持续供数时，才启用端到端tile DMA流水。
- 输入、两个中间通道和输出均使用有限FIFO/PIPO并进行死锁验证。

## 通信与内存

- Jetson CPU和iGPU共享物理DRAM，但缓存、映射和同步仍有成本。
- 比较CUDA device memory、pinned/registered host memory及驱动可供FPGA DMA访问的缓冲。
- 不预设GPUDirect或直接访问CUDA指针可用；由PCIe控制器、IOMMU和驱动测试决定。
- 权重应驻留FPGA片上存储或PL DDR，不能在每次请求中重复跨PCIe传输。
- 小tile不能逐个提交DMA，应使用持续流式描述符或只在FPGA内部切tile。

## 调度与验证

区域划分先比较强GPU成本和FPGA计算加通信成本；运行时再根据设备队列和链路状态计算
预计完成时间。若FPGA、DMA或缓冲条件不满足，自动退回最佳GPU engine。论文至少报告：
强GPU-only、完整FFN非流式、FPGA内部tile流、端到端tile流、自动策略，以及融合、Graph、
辅助流、DMA路径、tile大小、单双缓冲和引擎复制的消融。

当前能完成的是模型、实验矩阵、门禁和上板脚本。真实TensorRT engine、CUDA Graph、
Nsight时间线、HLS/RTL流水和DMA驱动必须在Jetson-RF880环境完成，不能由模拟代替。
