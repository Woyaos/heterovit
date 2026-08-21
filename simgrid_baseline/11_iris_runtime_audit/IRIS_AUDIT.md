# IRIS Runtime Audit for Jetson-FPGA ViT Inference

Audit date: 2026-07-30

## Decision

IRIS is suitable as the future real-hardware runtime baseline, but it is not a
simulator and it is not a complete communication-aware scheduler for the
target CUDA-GPU plus OpenCL-FPGA system. SimGrid remains the simulation engine.
The simulation should reproduce IRIS concepts so that a policy validated in
simulation can later be implemented as an IRIS custom policy.

## What IRIS Already Provides

| Capability | Status | Source evidence |
| --- | --- | --- |
| Heterogeneous task runtime | Yes | `README.md` describes simultaneous OpenMP, CUDA, HIP, Level Zero, OpenCL, and Hexagon backends. |
| Task dependencies and managed movement | Yes | `docs/sphinx/source/started.rst` documents data memory and scheduling policies. |
| NVIDIA GPU backend | Yes | `src/runtime/DeviceCUDA.cpp`. |
| Xilinx FPGA backend | Yes, through OpenCL/xclbin | `src/runtime/DeviceOpenCL.cpp:38-46`, `75-77`, and `671-740`. |
| Built-in locality policy | Yes, limited | `src/runtime/PolicyData.cpp` selects the device owning the largest input byte total. |
| Built-in profiling policy | Yes, compute-only selection | `src/runtime/History.cpp:98-112` selects the lowest average kernel time. |
| User-defined policy | Yes | `include/iris/iris_runtime.h:523` and `apps/custom_policy/PolicyGWS.cpp`. |
| Performance simulation without hardware | No | IRIS executes backend kernels on discovered devices; it does not emulate their timing. |

The local source is release 3.0.0 dated 2025-05-30 according to
`CHANGELOG.md`. It is therefore recent enough to study; project age is not the
reason to reject it.

## GPU-FPGA Communication Path

IRIS chooses direct device-to-device movement only when source and destination
have the same IRIS device type and both report D2D support
(`src/runtime/Device.cpp:976-990`). CUDA peer copies are implemented for CUDA
devices (`src/runtime/DeviceCUDA.cpp:667-710`). A Xilinx OpenCL accelerator is
classified as `iris_fpga`, while the NVIDIA device is a GPU, so this generic
test does not establish CUDA-to-OpenCL FPGA peer transfer.

When peer D2D is unavailable and the host copy is stale, IRIS falls back to:

1. source device to host memory;
2. host memory to destination device.

This fallback is visible in `src/runtime/Device.cpp:1420-1500` and
`src/runtime/Device.cpp:1949-1959`. The OpenCL buffer read/write calls inspected
in `src/runtime/DeviceOpenCL.cpp:323-383` use blocking `CL_TRUE`. IRIS 3.0 adds
asynchronous streams generally, but the useful overlap for a CUDA-to-Xilinx
path is backend- and memory-path-dependent and must not be assumed.

Jetson has an integrated GPU, so the eventual hardware experiment must test
the real memory route rather than model it as an ordinary discrete GPU. The
simulator must represent at least these alternative routes:

- staged copy through a host buffer;
- pinned/mapped shared host memory used by CUDA and FPGA DMA;
- a direct peer/RDMA route only if the selected Jetson and FPGA stack supports it.

## Scheduling Gap

IRIS records kernel and transfer histories, including H2D, D2D, and staged
D2H-H2D times (`src/runtime/History.cpp`). However, `OptimalDevice` compares
only average kernel time. It can therefore select FPGA because its Linear
kernel is faster even when transfer and queue delays make end-to-end completion
slower.

The `iris_data` policy is also not a time model. It selects the device owning
the largest aggregate number of bytes, and its source reports that DMEM is not
yet supported by this policy.

This gives a concrete systems research target: select or reject FPGA offload
using predicted finish time that includes compute, input movement, output
movement, queue delay, data residence, and available overlap.

## Required Baselines

The common ViT DAG should compare the following policies under the same cost
inputs:

1. GPU-only.
2. Static all-Linear FPGA offload.
3. IRIS-profile-like: choose by kernel time only.
4. IRIS-data-like: choose by current data residence only.
5. Communication-aware earliest-finish-time: compute plus transfers and queues.
6. Proposed policy: earliest finish time plus selective offload, activation
   residence, batching, and cross-request overlap.

The study must report single-request latency and steady-state throughput
separately. A ViT request has strict dependencies around most Linear layers,
so communication is difficult to hide inside one request. Pipelining different
requests is more likely to improve throughput than single-image latency.

## Next Implementation Milestone

Build one full-DAG SimGrid runner from the existing 125-task ViT graph. Its
platform model must expose separate GPU compute, FPGA compute, PCIe/DMA engines,
host-memory staging, direction-specific bandwidth/latency, and independent
queues. The runner must implement baselines 1-5 before adding the proposed
policy.

No absolute speedup claim is valid until costs are calibrated. Before hardware
arrives, results are conditional sensitivity and break-even results. After
hardware arrives, replace the same cost-table fields with microbenchmark
measurements and rerun the unchanged policies.
