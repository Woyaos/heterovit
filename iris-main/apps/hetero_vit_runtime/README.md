# Heterogeneous ViT IRIS Runtime Skeleton

This application is the hardware-independent bridge from the SimGrid study to
IRIS. `PolicyHeteroEFT.cpp` is an IRIS custom policy that reads eight integer
metadata fields from each task and combines calibrated GPU time, FPGA time,
input/output communication, queue depth, and latency/throughput objective.

`generate_manifest.py` converts the fused ViT DAG and a calibrated profile into
a C header. `vit_dag_skeleton.c` creates the complete dependency graph, attaches
the metadata, registers the custom policy, and submits every task. The included
OpenMP no-op kernel permits a CPU-only smoke test. It validates IRIS loading,
metadata, DAG dependencies, and policy fallback but does not validate GPU-FPGA
placement or performance.

Generate the manifest from the repository root:

```bash
python3 iris-main/apps/hetero_vit_runtime/generate_manifest.py \
  simgrid_baseline/14_optimization_ablation/results/dags/ffn_fused.json \
  simgrid_baseline/14_optimization_ablation/results/profiles/fusion_compression_one_copy_double_buffer.json \
  iris-main/apps/hetero_vit_runtime/vit_task_manifest.h
```

After IRIS is installed, build and run the CPU-only smoke test. The explicit
environment variables avoid depending on a shell-specific setup script:

```bash
cd iris-main/apps/hetero_vit_runtime
make IRIS=$PWD/../../install-cpu
./test_cost_model
LD_LIBRARY_PATH=$PWD:$PWD/../../install-cpu/lib \
IRIS_ARCHS=openmp \
KERNEL_BIN_OPENMP=$PWD/kernel.openmp.so \
./vit_dag_skeleton --latency
```

Use `--throughput` to reduce the communication penalty and favor sustained
FPGA occupancy. In the complete runtime, the request-level backlog controller
selects between these two objectives; it does not switch individual operators
independently without regard to the request SLO.

On target hardware, provide CUDA and Xilinx implementations of the named fused
subgraphs, replace the no-op tasks with real memory objects and kernels, update
the manifest from measured costs, and run with `IRIS_ARCHS=cuda,opencl,openmp`.
