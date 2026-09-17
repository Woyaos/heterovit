# Communication-Expansion-Aware ViT Offloading

## English manuscript section

### A. Structural Motivation and Fusion Bound

On the board-level PCIe x4 Jetson AGX-RFSoC platform, faster low-bit Linear kernels can still increase end-to-end ViT latency because device crossings have fixed and payload-dependent costs. We first locate avoidable crossings in the ONNX-derived DAG. It contains twelve branch-free `FC1 -> GELU -> FC2` feed-forward (FFN) chains with GPU-resident entry and exit; surrounding attention and residual dependencies remain intact.

Let the FFN input occupy \(S\) bytes and each expanded intermediate occupy \(rS\) bytes at the same precision. With both FCs on FPGA but GELU on GPU, four crossings move \(2(1+r)S\) logical bytes. A complete FPGA FFN keeps the expanded intermediates local and moves \(2S\) bytes across two crossings. Fusion therefore saves \(r/(1+r)\) of boundary bytes and two fixed transfer costs. In the actual FP16 DAG, \(S=302{,}592\) B and \(r=4\): boundary volume falls 80% (3.026 to 0.605 MB per FFN). This is neither an 80% latency prediction nor proof of sufficient on-chip memory.

For crossing edge \(e\), we use \(\tau_e=\kappa_e+\sum_{h\in\mathcal P_e}(\alpha_h+s_e/B_h)\). The conservative host-staged path includes GPU-host and host-FPGA PCIe stages; \(\kappa_e\) accounts for format handling. With computation and dispatch costs \(C_{\rm fused}\) and \(C_{\rm split}\), the complete FFN beats the split mapping under an isolated additive model iff

\[
C_{\rm fused}-C_{\rm split}<\tau_{F\to G}(rS)+\tau_{G\to F}(rS). \tag{1}
\]

It beats GPU-only iff \(C_G-C_{\rm fused}>\tau_{G\to F}(S)+\tau_{F\to G}(S)\). These are conditional benefits, not a global optimality claim over arbitrary ViT subgraphs.

### B. Executable Mapping and System Evidence

FPGA blocks require declared operator support, matching precision, and sufficient conservative workspace. A complete FFN additionally requires FPGA GELU and intermediate storage; otherwise only supported singleton FCs are legal. A device/block dynamic program minimizes additive compute and boundary cost for each certified serial chain with GPU-fixed interfaces. The resulting chain optimum excludes shared-resource queuing and branched-DAG effects. We retain external edges while contracting selected blocks, then replay the full DAG in SimGrid with shared compute and transfer resources. The planned device is exported to IRIS; its CPU-only skeleton currently performs no ViT arithmetic.

Under an **unmeasured sensitivity profile**, single-request SimGrid times are 24.749 ms (GPU-only), 31.129 ms (split FFN Linear offloading), and 17.994 ms (complete FFN). The latter reduces twelve-FFN boundary bytes from 36.311 to 7.262 MB. The automatic plan equals fixed full-FFN offloading under uniform costs; without an assumed full-FFN kernel, it selects GPU-only. These results validate the modeled mechanism. Hardware claims require actual FPGA kernels, PCIe characterization, and held-out end-to-end measurements.

## 中文对照

### A. 结构瓶颈与融合收益边界

在设计为通过 PCIe x4 互连的 Jetson AGX-RFSoC 平台上，低比特 Linear 内核加速不一定缩短整个 ViT 的推理时间。每次跨设备传输都有固定开销，传输激活值也需要时间；两者可能抵消内核计算收益。因此，我们先分析 ViT 任务图中哪些跨设备传输可以避免，再决定哪些算子应当卸载。由 ONNX 提取的任务图包含 12 个没有额外分支的 `FC1 -> GELU -> FC2` 前馈网络（FFN）链；其入口和出口留在 GPU，从而保留周围的 Attention 与残差依赖。

设 FFN 输入激活占 \(S\) 字节，中间激活在相同精度下占 \(rS\) 字节。若两个 FC 在 FPGA 而 GELU 在 GPU，数据跨设备四次，逻辑传输量为 \(S+rS+rS+S=2(1+r)S\)。若整条 FFN 链在 FPGA 执行，扩大的两个中间结果留在 FPGA 本地，只传入输入和传回输出，总量为 \(2S\)。因此，融合减少两次传输，逻辑跨设备数据量降低 \(r/(1+r)\)，这一结构结论不依赖所假设的 PCIe 带宽。在当前任务图的 FP16 精度下，\(S=302{,}592\) 字节、\(r=4\)，每个 FFN 的数据量从 3.026 MB 降为 0.605 MB，即减少 80%。这只是**传输字节数**的结论，不表示推理延迟降低 80%，也不证明 FPGA 的片上存储足够。

跨设备边 \(e\) 的传输成本记为 \(\tau_e=\kappa_e+\sum_{h\in\mathcal P_e}(\alpha_h+s_e/B_h)\)。其中 \(\mathcal P_e\) 是假设的数据路径，\(\alpha_h\) 是每段固定时间，\(B_h\) 是有效带宽，\(\kappa_e\) 包括数据格式处理。目前保守采用 GPU-主存和主存-FPGA PCIe 两段中转路径。令 \(C_{\rm fused}\) 和 \(C_{\rm split}\) 包含内核计算与任务启动，则在独立、成本可加的区域模型中，整条 FFN 在 FPGA 执行优于分离执行，当且仅当

\[
C_{\rm fused}-C_{\rm split}<\tau_{F\to G}(rS)+\tau_{G\to F}(rS). \tag{1}
\]

整条 FFN 优于 GPU-only，还要求其计算收益超过剩余的输入、输出两次传输成本。以上条件说明 FFN 融合**在什么情况下**值得采用；它不证明 FFN 是 ViT 所有子图中收益最大的区域。

### B. 可执行的映射与系统证据

模型只有在已声明的 FPGA 算子支持、激活精度和保守工作空间上限均满足时，才接纳 FPGA 执行块。完整 FFN 需要 FPGA 侧 GELU 和中间激活存储；缺少这种内核时，只考虑实际声明支持的单个 FC。对每条经过依赖检查的串行 FFN，在 GPU 固定入口和出口的条件下，设备/块动态规划最小化块计算与边界传输成本。这个最优性只针对该链及其可加的成本目标；整图中的共享资源等待和残差分支另在完整任务图上评估。选定执行块后保留全部外部依赖，在 SimGrid 中按共享计算设备和传输链路进行事件回放。计划设备也导出到 IRIS 清单；当前 CPU-only 骨架还不执行 ViT 计算。

在明确标注为**未实测敏感性参数**的配置下，单请求 SimGrid 时间分别为 GPU-only 24.749 ms、两个 FFN Linear 分离卸载 31.129 ms、完整 FFN 卸载 17.994 ms。12 个 FFN 的逻辑传输量从 36.311 MB 降为 7.262 MB；当前均匀成本配置中自动计划与固定完整 FFN 卸载相同。若从 FPGA 假设能力中移除完整 FFN/GELU 内核，策略退回 GPU-only。这些实验验证了设计机制和可行性检查，不能证明目标板卡已有加速。仍需用实际 FPGA 内核、PCIe 建链和 Jetson/FPGA/传输实测数据校准和验证模型，才能报告硬件性能结论。

## 写作依据与证据定位（不纳入论文正文）

| 段落 | 为什么这样写 | 当前证据 |
| --- | --- | --- |
| A1 | 先提出真实部署决策问题，再说明分析对象来自任务依赖，而非先假定 FFN 最优 | `08_coarse_vit_dag/results/vit_batch1_coarse_dag.json`、`partition.py` |
| A2 | 用 ViT FFN 扩展比推导可避免的中间激活传输；给出适用边界 | `structural_evidence.py`、`results/structural_evidence.json` |
| A3 | 把固定通信开销、带宽和 FPGA GELU 计算代价纳入条件判断；排除无条件融合主张 | `12_full_dag_baselines/run_policy.py`、`results/structural_evidence.json` |
| B1 | 交代能否实现、如何映射、区域最优与整图评估的关系 | `partition.py`、`test_partition.py`、IRIS planned manifest |
| B2 | 给出同条件三方对照、能力退化和证据等级；不把模拟写成真实板上加速 | `results/structural_replay/`、`test_structural_evidence.py` |

文献定位：EQ-ViT 的可借鉴之处是逐内核分析、针对性架构设计和真实板上验证的连续证据链，不是本节传输量公式的来源。SimGrid 官方文档说明它依据输入的平台参数和资源活动计算事件时间；其网络模型必须校准。当前本机 SimGrid 3.35 使用 CM02 而非适于 PCI 类总线的 Raw 模型，因此本章只报告未校准敏感性结果。

- EQ-ViT 原文：https://peipeizhou-eecs.github.io/publication/2024_esweek_eqvit/2024_esweek_eqvit.pdf
- SimGrid 官方模型说明：https://simgrid.org/doc/latest/Models.html
- ISCAS 2026 模板：https://epapers2.org/iscas2026/ESR/samples.php

正文英文约 400--450 词，实际是否占一页需要以最终 IEEE 双栏模板编译排版为准；图表和参考文献编号应在全文统一安排。
