# SimGrid Experiment Log

Environment: Windows WSL 2, Ubuntu 24.04.1 LTS, Python 3.12.3, SimGrid 3.35.

All times below are simulated times. Device speeds and PCIe parameters are
teaching values, not measurements from the target Jetson-FPGA platform.

## Experiment 01: One Virtual Device

- Goal: verify that SimGrid can advance virtual compute time.
- Configuration: one 1 Gf/s virtual GPU; 10 million operations.
- Command: `python3 01_hello/hello_simgrid.py 01_hello/platform.xml`
- Result: 10.000 ms makespan.
- Conclusion: compute duration follows work divided by virtual device speed.
- Limitation: no communication, heterogeneity, or contention.

## Experiment 02: One GPU-FPGA Pipeline

- Goal: verify GPU compute, PCIe transfer, FPGA compute, return transfer, and final GPU compute.
- Configuration: GPU 1 ms; 4 MB transfer; FPGA 0.4 ms; 4 MB return; GPU 1 ms; PCIe 4 GB/s and 20 us latency.
- Command: `python3 02_gpu_fpga_pipeline/pipeline.py 02_gpu_fpga_pipeline/platform.xml`
- Result: 4.540 ms makespan with the CM02 network model.
- Conclusion: a complete offload path includes both computation and two transfers.
- Limitation: one serial request with synthetic parameters.

## Experiment 03: GPU-Only vs FPGA Offload

- Goal: show that the best placement depends on PCIe communication cost.
- Configuration: Linear is 2 ms on GPU and 0.4 ms on FPGA; input and output are both 4 MB.
- Command: `python3 03_strategy_comparison/compare.py`
- Results: at 4 GB/s, GPU-only 4.000 ms and offload 4.540 ms; at 16 GB/s, GPU-only 4.000 ms and offload 2.965 ms.
- Conclusion: FPGA kernel speed alone is insufficient; total path latency determines the best strategy.
- Limitation: strategies are evaluated separately, not selected by an online scheduler.

## Experiment 04: Shared Resource Contention

- Goal: compare one request with two simultaneous offload requests sharing one GPU, one FPGA, and one PCIe link.
- Configuration: same offload path as Experiment 02; both requests arrive at time zero.
- Command: `python3 04_shared_resource_contention/compare.py`
- Results: one request takes 4.540 ms; two concurrent requests take 9.040 ms versus a 9.080 ms sequential reference. Throughput changes from 220.264 to 221.239 requests/s.
- Conclusion: simultaneous submission alone provides almost no throughput gain because the requests contend in lockstep for every shared resource.
- Limitation: SimGrid shares host capacity between simultaneous activities; the model does not yet represent measured GPU kernel concurrency, FPGA pipeline initiation intervals, or staggered arrivals.

## Experiment 05: Identify the ONNX Workload

- Goal: determine the actual architecture represented by `vit.onnx` before using it as a scheduling workload.
- Configuration: read-only ONNX graph inspection in the `hetero_vit` environment; no target-device timing measurements.
- Command: `conda run -n hetero_vit python simgrid_baseline/05_vit_workload/inspect_vit_onnx.py vit.onnx simgrid_baseline/05_vit_workload/results`.
- Results: input `[batch,3,224,224]`; output `[batch,1000]`; 86,453,224 parameters; hidden dimension 768; 12 Transformer blocks; 49 constant-weight Linear nodes.
- Linear shapes: twelve each of `768x2304`, `768x768`, `768x3072`, and `3072x768`, plus one `1000x768` classifier.
- Conclusion: the existing file is ViT-Base/16-like, not ViT-Small. It will be used as the initial systems workload because its graph has been verified.
- Limitation: this is an image-classification ViT rather than the final semantic-communication model, and no Jetson, FPGA, or PCIe timings are available yet.

## Experiment 06: Linear Activation Tensor Sizes

- Goal: calculate the runtime data transferred when each Linear is individually offloaded at batch 1.
- Configuration: ONNX shape inference plus verified ViT dimensions; FP32 activations; weights assumed resident in FPGA-attached memory before inference.
- Command: `conda run -n hetero_vit python simgrid_baseline/06_linear_tensor_sizes/extract_linear_tensors.py vit.onnx simgrid_baseline/06_linear_tensor_sizes/results/linear_tensor_sizes.csv`.
- Results: 49 candidates. Per-block QKV round trip is 2,420,736 bytes; attention projection is 1,210,368 bytes; FFN1 and FFN2 are each 3,025,920 bytes. The classifier round trip is 7,072 bytes.
- Conclusion: FFN offload saves substantial GPU computation but creates about 3.03 MB of activation traffic per isolated Linear at batch 1, so grouping and activation precision will materially affect the break-even point.
- Limitation: values are FP32 tensor sizes, not measured transfer times. Quantized activation formats, padding, DMA alignment, and data-layout conversion are not included.

## Experiment 07: Raw ONNX Task DAG

- Goal: convert the verified ONNX model into a device-independent operator DAG for later scheduling experiments.
- Configuration: batch fixed to one in an in-memory model copy; ONNX shape inference; FP32 tensor sizes; weights treated as preloaded device data.
- Command: `conda run -n hetero_vit python simgrid_baseline/07_onnx_task_dag/build_task_dag.py vit.onnx simgrid_baseline/07_onnx_task_dag/results`.
- Results: 1,003 ONNX nodes, including 740 runtime nodes and 263 Constant nodes; 1,151 tensor-dependency edges; 49 GPU/FPGA Linear candidates; all edge byte sizes resolved; DAG acyclicity check passed.
- Conclusion: the model now has a reproducible dependency representation with real batch-1 tensor sizes and explicit device eligibility, without assigning fabricated device times.
- Limitation: raw ONNX granularity contains shape-management and other fine-grained operators that should not be scheduled independently. A coarse task graph is required before adding cost models or schedulers.

## Experiment 08: Coarse ViT Scheduling DAG

- Goal: merge fine-grained ONNX operators into practical scheduling tasks while preserving Transformer residual dependencies.
- Configuration: two embedding tasks; ten tasks per Transformer block; final normalization, class-token selection, and classifier; batch-1 FP32 edge sizes.
- Command: `conda run -n hetero_vit python simgrid_baseline/08_coarse_vit_dag/build_coarse_dag.py simgrid_baseline/07_onnx_task_dag/results/vit_batch1_raw_dag.json simgrid_baseline/08_coarse_vit_dag/results`.
- Results: 125 coarse tasks and 148 dependency edges; 49 GPU/FPGA Linear tasks and 76 GPU-only tasks; 646 runtime ONNX nodes grouped; 94 weight-alias Identity nodes ignored; no empty groups; acyclicity check passed.
- Conclusion: the resulting DAG is suitable as the common workload input for GPU-only, static-offload, EFT/HEFT, and future proposed schedulers. Residual1 and Residual2 each retain both their compute and skip inputs.
- Limitation: execution times are intentionally absent. Coarsening assumes each listed group is dispatched as one scheduling unit; later runtime implementation must use compatible subgraph boundaries.

## Experiment 09: Cost Data Interface

- Goal: define auditable target-device timing and communication inputs without inserting invented values.
- Configuration: one cost row per eligible task-device pair plus a target platform profile for GPU, FPGA, PCIe directionality, and overlap support.
- Commands: generate with `generate_cost_templates.py`; inspect completeness with `validate_cost_data.py`; use `--strict` before a formal simulation.
- Results: 174 cost rows generated from 125 tasks: 125 GPU rows and 49 FPGA rows. All 174 timing rows and 19 required platform fields are currently incomplete, so validation reports `valid_for_simulation=false`; strict validation rejects execution.
- Conclusion: model structure and tensor sizes are now cleanly separated from unavailable device measurements. Future Jetson, FPGA, HLS, and PCIe data have explicit source, hardware, sample-count, and percentile fields.
- Limitation: no target hardware is currently available, so this experiment creates and validates the interface only; it does not produce a latency prediction.

## Experiment 10: Break-Even Sensitivity Sweep

- Goal: derive conditional offload boundaries without inventing target GPU execution times.
- Configuration: FPGA speedup over GPU in `{2,4,8}`; effective PCIe bandwidth in `{2,4,8,16}` GB/s; one-way fixed latency in `{10,20,50}` us; activation precision in `{FP32,FP16,INT8}`; preloaded FPGA weights; no overlap, queueing, or contention.
- Command: `conda run -n hetero_vit python simgrid_baseline/10_break_even_sensitivity/run_break_even_sweep.py simgrid_baseline/08_coarse_vit_dag/results/vit_batch1_coarse_dag.json simgrid_baseline/10_break_even_sensitivity/sensitivity_config.json simgrid_baseline/10_break_even_sensitivity/results`.
- Results: 540 conditional rows across five unique Linear types. At FP32, 4 GB/s, 20 us per direction, and 4x FPGA compute speedup, minimum GPU latency for isolated offload is 860.245 us for QKV, 456.789 us for projection, 1,061.973 us for each FFN Linear, and 55.691 us for the classifier.
- Precision result: under the same FFN1 condition, the break-even GPU latency falls from 1,061.973 us at FP32 to 557.653 us at FP16 and 305.493 us at INT8 because activation traffic decreases.
- Conclusion: real Jetson timings can later be compared directly with these thresholds. The sweep identifies conditions under which isolated offload could win; it does not predict that those conditions hold on the target platform.
- Limitation: isolated tasks only; continuous Linear grouping, computation/communication overlap, multiple requests, resource contention, layout conversion, and host-staged PCIe copies are excluded.

## Experiment 11: IRIS Runtime and Communication Audit

- Goal: decide whether IRIS should be the simulator, the future hardware runtime, or neither, and identify the exact scheduling gap relevant to Jetson-GPU and PCIe-FPGA inference.
- Configuration: static source audit of the local `iris-main` release; no target hardware and no simulated timing parameters.
- Evidence: IRIS 3.0.0 (2025-05-30) supports CUDA, OpenCL/Xilinx FPGA binaries, task dependencies, managed data movement, profiling, asynchronous facilities, and loadable custom policies. Direct D2D selection requires matching IRIS device types; the generic cross-type path stages data through host memory. The built-in profile policy selects by average kernel time without adding transfer time, although transfer histories are recorded.
- Result: IRIS is accepted as the future real-hardware runtime baseline, not as the simulator. SimGrid remains the modeling engine, and its scheduler/runtime abstractions will be aligned with IRIS so that the proposed policy can later be ported as an IRIS custom policy.
- Research gap: compare compute-only and locality-only IRIS-style decisions with communication-aware earliest-finish-time scheduling that includes input/output transfer, queue delay, residence, and overlap.
- Artifact: `11_iris_runtime_audit/IRIS_AUDIT.md`.
- Limitation: source behavior has been audited but IRIS has not yet been built or executed against CUDA plus Xilinx OpenCL hardware. Jetson pinned memory, zero-copy, and direct DMA support depend on the eventual board and driver stack and must be measured.

## Experiment 12: Full ViT DAG Baseline Policies

- Goal: execute the complete coarse ViT DAG and compare GPU-only, static FPGA offload, IRIS-profile-like, IRIS-data-like, and communication-aware placement under identical resource assumptions.
- Configuration: 125 tasks and 148 edges; one inference request; FP32 activations; FPGA weights preloaded; every GPU-FPGA dependency is explicitly replayed as device-to-HOST followed by HOST-to-device. Both input profiles declare `profile_kind=sensitivity` and `valid_for_target_prediction=false`.
- Reference conditional point: GPU matmul 2 effective TFLOP/s, FPGA Linear 4x compute speedup plus 8 us dispatch, GPU-HOST 20 GB/s plus 5 us, HOST-FPGA 4 GB/s plus 20 us. GPU-only took 22,184.376 us. Static all-Linear FPGA took 48,870.450 us, and IRIS-profile-like took 48,840.809 us because kernel-only decisions ignored the staged return path. Communication-aware EFT rejected all offloads and matched GPU-only.
- Favorable conditional point: FPGA Linear 8x compute speedup plus 3 us dispatch, GPU-HOST 40 GB/s plus 2 us, HOST-FPGA 16 GB/s plus 5 us. GPU-only took 22,184.376 us; static all-Linear took 18,830.446 us; communication-aware EFT took 18,559.331 us, a conditional 1.195x speedup.
- Selective placement result: communication-aware EFT offloaded 36 tasks in the favorable point: all QKV, FFN1, and FFN2 Linears, while retaining all projection Linears and the classifier on GPU. It used 72 cross-device edges instead of the static policy's 97.
- Validation: every policy produced exactly 125 task records; every cross-device edge produced exactly two transfer-stage records; GPU-only analytical and SimGrid makespans matched to floating-point precision; all summaries remain invalid for target prediction.
- Artifacts: `12_full_dag_baselines/RESULTS.md`, `results/policy_comparison.csv`, and `results_favorable/policy_comparison.csv`, with per-policy placements and task/transfer timelines.
- Conclusion: communication-aware selective offload behaves correctly in both an unprofitable and a profitable region, while compute-only IRIS-style placement can be badly wrong when host-staged communication dominates. This establishes the baseline mechanism, not a Jetson-FPGA speedup claim.
- Limitation: single request only; sensitivity rather than measured costs; no cross-request overlap, pinned/zero-copy path, direct DMA, layout conversion, power, or hardware calibration.

## Experiment 13: Communication Design Space and Multi-Request Pipeline

- Goal: finish the hardware-free study with one integrated experiment covering communication-path stages, bandwidth, fixed latency, FPGA compute speedup, selective placement, and cross-request pipeline behavior.
- Design space: three path abstractions (`host_staged_two_copy`, `shared_mapped_one_copy`, and `direct_dma_one_copy`), effective bandwidth in `{2,4,8,16}` GB/s, one-way fixed latency in `{5,10,20,50}` us, and FPGA Linear speedup in `{2,4,8,16}`. The sweep contains 192 hardware conditions and 576 policy rows.
- Path result: communication-aware EFT was profitable in 6 of 64 host-staged conditions, versus 19 of 64 matched one-copy conditions. Maximum conditional EFT speedup was 1.105x for host staging and 1.541x for one-copy paths.
- Host-staged threshold: no tested bandwidth below 16 GB/s was profitable. At 16 GB/s, profitable points required at least 8x FPGA Linear speedup and no more than 20 us fixed latency. No tested host-staged point at 50 us was profitable.
- Best analytical point: a matched one-copy path at 16 GB/s, 5 us, and 16x FPGA Linear speedup produced a conditional 1.541x speedup with 48 FPGA Linears. This is a sensitivity boundary, not a hardware prediction.
- Pipeline replay: four representative profiles were replayed in SimGrid with `{1,2,4,8}` simultaneous requests and three policies, producing 48 rows. In the favorable staged profile, communication-aware placement used 36 FPGA Linears and reached 129.918 requests/s at eight requests versus GPU-only at 45.077 requests/s, a conditional 2.882x throughput speedup.
- Objective conflict: at the matched one-copy midpoint, static offload had 0.851x single-request performance but 1.435x throughput at eight requests. Latency-oriented EFT retained GPU-only, proving that low-latency and high-throughput scheduling must be reported as different objectives.
- Validation: `validation_report.json` passes all 12 checks. The 48 replay rows contain the expected task, completion, and transfer records, and every output declares `valid_for_target_prediction=false`.
- Artifacts: `13_design_space_and_pipeline/RESULTS.md`, `results/design_space_sweep.csv`, `results/bandwidth_thresholds.csv`, `results/pipeline_replay.csv`, and `results/validation_report.json`.
- Conclusion: the hardware-free simulator, baseline policies, communication-path abstractions, threshold study, multi-request replay, validation, and calibration interface are complete. Further numerical refinement requires measured target-hardware costs.
- Limitation: the matched shared-memory and direct-DMA abstractions intentionally produce equal timing when assigned equal bandwidth and latency. Real cache synchronization, mapping, driver, DMA, layout, power, and thermal effects remain unmeasured.
## Experiment 14: Communication Optimization Ablation

- Goal: determine whether activation compression, fused FPGA FFN residency, one-copy communication, and double buffering can change the reference-point result.
- Configuration: Experiment 12 reference hardware assumptions; original and fused ViT DAGs; no compression, 2x compression, and 4x compression; host-staged, shared-mapped, and direct-DMA path abstractions; one or two FPGA input buffers; one and eight simultaneous requests.
- Results: the original communication-aware policy remains GPU-only with compression or one-copy communication alone. Fusing each `FFN1 + GELU + FFN2` island selects 24 Linear equivalents and improves single-request time from 22.184 ms to 18.910 ms. Adding 2x compression reaches 17.299 ms, and adding a one-copy path reaches 16.798 ms, a conditional 1.321x speedup. For eight requests, the full-stack communication-aware case reaches 102.227 requests/s or 2.268x GPU-only throughput. Static all-offload with two input buffers reaches 105.952 requests/s or 2.350x, showing that latency and throughput require different placement objectives.
- Validation: 54 replay rows were generated and all 14 automated checks passed, including legacy baseline regression, DAG shape, record counts, compression monotonicity, buffer behavior, and shared/direct equivalence under matched abstract costs.
- Conclusion: reducing device transitions through fused FFN residency is more decisive than compression or one-copy communication alone at the reference point. Compression and path optimization become valuable when combined with fusion, while double buffering primarily benefits saturated throughput.
- Limitation: codec cost, FPGA GELU cost, mapped-memory synchronization, direct-DMA feasibility, and buffer overlap are sensitivity assumptions rather than target measurements.

## Experiment 15: Dynamic Dual-Mode Runtime

- Goal: replace simultaneous batch replay with request arrivals, queueing, SLO metrics, and request-level switching between latency and throughput objectives.
- Configuration: 32 requests per case; offered rates `{20,40,60,80,100,120}` requests/s; 30 ms SLO; device-serialized activation codec work; fused FFN DAG; full-stack sensitivity profile; adaptive backlog thresholds `{1,2,4,8}`.
- Results: 24 load cases and 12 threshold cases completed. Threshold 4 was selected. At 20 requests/s, latency EFT reduced P95 from GPU-only 22.184 ms to 16.798 ms. At 80 requests/s, GPU-only P95 was 417.533 ms while adaptive P95 was 24.330 ms with no SLO violations. At 120 requests/s, adaptive used 6 latency and 26 throughput requests, reached 108.044 requests/s, and had 83.677 ms P95; the system was overloaded and 93.75% of requests violated the SLO.
- Validation: `validate_results.py` passed all structural checks over 24 load and 12 threshold cases. Every output remains `valid_for_target_prediction=false`.
- Conclusion: a single latency policy is insufficient once the system saturates. Request-level mode selection can improve overloaded throughput and tail latency, but it cannot make offered load above capacity satisfy the SLO.
- Artifacts: `15_dynamic_runtime/RESULTS_CN.md`, `results/dynamic_load_sweep.csv`, task/transfer timelines under `results/rate_*`, and `results_thresholds/threshold_sweep.csv`.

## IRIS Runtime Bridge

- Goal: port the communication-aware cost decision from simulation into a real heterogeneous runtime interface before hardware is available.
- Implementation: `iris-main/apps/hetero_vit_runtime` contains a loadable `hetero_eft` policy, eight task metadata fields, a generated 101-task/124-edge ViT manifest, latency/throughput objective selection, an OpenMP no-op backend, and a cost-model unit test.
- Validation: IRIS 3.0 was built from the local source with OpenMP support. The complete DAG ran successfully through the custom policy in both `--latency` and `--throughput` modes on the CPU-only fallback backend.
- Limitation: CUDA/FPGA placement, real kernels, managed data movement, and measured performance require the target hardware and drivers.

## Experiment 16: Target Hardware Calibration Handoff

- Goal: make the transition from sensitivity assumptions to auditable target measurements mechanical and reject incomplete data.
- Results: generated a fused-DAG plan with 138 task/device rows: 101 GPU and 37 FPGA. Generated 60 transfer rows covering 10 payload sizes and six directions for one-copy, direct, and host-staged paths. The largest planned payload is 4,194,304 bytes.
- Interface: `validate_and_build_profile.py` requires measured mean/P50/P95, at least 30 samples, hardware and source provenance, complete platform state, and measured transfer curves. It fits directional fixed latency and bandwidth and builds a profile accepted by SimGrid and the IRIS manifest generator.
- Current status: validation intentionally fails with 138 task/device rows, all required transfer points, and platform fields incomplete. This is the first point at which target Jetson and FPGA hardware are genuinely required.
