# Communication-Aware FFN Partitioning

## A. Dependency-Constrained Offloading

Low-bit acceleration reduces arithmetic cost, but repeated device crossings can offset this benefit in PCIe-connected GPU-FPGA inference. Motivated by the computation/data-movement co-design in [EQ-ViT](https://peipeizhou-eecs.github.io/publication/2024_esweek_eqvit/2024_esweek_eqvit.pdf), we formulate offloading at the subgraph level. Our focus is jointly selecting executable FFN blocks and their device assignments under explicit communication costs.

Let the inference graph be a directed acyclic graph \(G=(V,E)\), with tensor size \(s_e\) on edge \(e\). We identify serial FFN regions \(\mathrm{FC}_1\rightarrow\mathrm{GELU}\rightarrow\mathrm{FC}_2\) whose internal tensors have no external consumers and whose entry and exit remain on the GPU. Residual and attention dependencies are preserved outside each region. An FPGA candidate is admitted only when its operator sequence, precision, and conservative activation footprint satisfy a declared kernel-capability and memory budget. A complete FFN candidate assumes FPGA-side GELU and intermediate storage; otherwise, only supported singleton operators are considered.

## B. Boundary Cost and Offloading Criterion

For devices \(p,q\in\{G,F\}\), define

\[
\tau_e(p,q)=\begin{cases}
0,&p=q,\\
\kappa_e+\displaystyle\sum_{\ell\in\mathcal P_{pq}}
\left(\alpha_\ell+\frac{s_e}{B_\ell}\right),&p\ne q,
\end{cases}\tag{1}
\]

where \(\mathcal P_{pq}\) is the physical transfer path, \(\alpha_\ell\) and \(B_\ell\) denote fixed latency and effective bandwidth, and \(\kappa_e\) includes encoding/decoding costs. Host-staged communication explicitly includes GPU-host and host-FPGA stages. Let \(C_d(b)\) include computation and dispatch for block \(b\) on device \(d\). The gain of complete-region FPGA offloading is

\[
\Delta_b=C_G(b)-C_F(b)-\tau_{\mathrm{in}}(G,F)
-\tau_{\mathrm{out}}(F,G).\tag{2}
\]

Under the isolated-region additive model, offloading improves latency exactly when \(\Delta_b>0\). This condition prevents compute-only decisions from accepting communication-dominated mappings. Keeping the expanded FFN activation on the FPGA removes internal device crossings; this benefit remains conditional on an executable complete-FFN implementation.

## C. Exact Regional Partitioning and System Evaluation

Building on communication-aware device selection as studied in [HEFT](https://doi.org/10.1109/71.993206), we also select block boundaries. For a region containing \(n\) operators, let \(F(j,d)\) be the minimum cost of its first \(j\) operators ending on device \(d\). For admissible blocks \([i,j)\),

\[
F(j,d)=\min_{i<j,\,p}
\left\{F(i,p)+\tau_{e_i}(p,d)+C_d([i,j))\right\}.\tag{3}
\]

Initialize \(F(0,G)=0\) and other infeasible states to infinity; the regional optimum is \(\min_d\{F(n,d)+\tau_{e_n}(d,G)\}\). Here \(e_0\) and \(e_n\) are the entry and exit edges. Every partition has a final block and an optimal prefix, so induction establishes optimality for the certified serial region. Exhaustive enumeration verifies the implemented recurrence. This guarantee excludes inter-region queuing and general branched-DAG scheduling.

Selected FPGA blocks are contracted into tasks while retaining external dependencies. We then evaluate the complete graph in SimGrid with shared compute/link resources and bounded FPGA input buffers. Under the assumed evaluation profile, single-request latency is 24.749 ms for GPU-only, 36.741 ms for singleton-Linear offloading, and 17.994 ms for regional partitioning. The default partition matches fixed full-FFN offloading. These conditional simulations support communication-aware granularity selection; target-hardware calibration remains necessary to establish deployment gains.

---

## 使用说明（不纳入英文正文）

- 正文面向 IEEE 双栏约一页的方法章节；实际页数需以论文模板编译结果为准。ISCAS 2026 规定技术内容最多四页，可另加一页仅用于参考文献，后续届次以对应官方要求为准：https://2026.ieee-iscas.org/authors/author-instructions.html
- 最终排版将两处论文链接替换为全文统一的 IEEE 编号引用；参考文献条目放在论文末尾。
- 核心论点：以合法、可执行的子图为候选，计算一次块计算及跨设备边通信，用区域 DP 联合决定粒度与设备，再用完整 DAG 事件仿真评价。
- 式 (2) 的充要条件仅限独立区域、可加成本、GPU 固定入口出口。式 (3) 的最优性仅限认证串行区域。不要扩写成任意 DAG 全局最优。
- 候选内核能力与存储预算目前为声明的设计假设。代码没有证明 RF880 已支持完整 FFN/GELU，也没有实现该 FPGA 内核。
- 模型成本参数、网络模型限制、12/9 组并发/在线结果、双缓冲和 IRIS 清单导出应留给实验/系统实现章节，避免挤占此节篇幅。
- 当前工作能够支撑方法与仿真章节；是否录用还取决于全文的新颖性、强基线、真实硬件证据及审稿。该节没有把动态规划、融合或双缓冲本身声称为首次提出。
