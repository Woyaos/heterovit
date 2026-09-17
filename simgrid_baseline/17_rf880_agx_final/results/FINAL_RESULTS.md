# RF880-AGX Final Experiment Summary

This run is a hardware-informed sensitivity experiment. It fixes the RF880 + Jetson AGX PCIe x4 topology, but it is not a target-hardware measurement until the generated measurement templates are filled with real data and converted into a complete target profile.

## Scope

- DAG: D:\VS_project\heterogeneous\simgrid_baseline\08_coarse_vit_dag\results\vit_batch1_coarse_dag.json
- Base profile: D:\VS_project\heterogeneous\simgrid_baseline\17_rf880_agx_final\rf880_agx_sensitivity_profile.json
- Output directory: D:\VS_project\heterogeneous\simgrid_baseline\17_rf880_agx_final\results
- Target validity: False

## Single-Request Ablation

| Case | Policy | Latency (ms) | Speedup vs same-case GPU | FPGA Linear equivalents | Cross-device edges |
| --- | --- | ---: | ---: | ---: | ---: |
| rf880_baseline_linear_host_staged | gpu_only | 24.749 | 1.000x | 0 | 0 |
| rf880_baseline_linear_host_staged | static_all_fpga | 35.591 | 0.695x | 49 | 97 |
| rf880_baseline_linear_host_staged | communication_eft | 24.749 | 1.000x | 0 | 0 |
| rf880_activation_compression_2x | gpu_only | 24.749 | 1.000x | 0 | 0 |
| rf880_activation_compression_2x | static_all_fpga | 27.385 | 0.904x | 49 | 97 |
| rf880_activation_compression_2x | communication_eft | 24.749 | 1.000x | 0 | 0 |
| rf880_ffn_fusion_residency | gpu_only | 24.749 | 1.000x | 0 | 0 |
| rf880_ffn_fusion_residency | static_all_fpga | 23.031 | 1.075x | 49 | 73 |
| rf880_ffn_fusion_residency | communication_eft | 17.850 | 1.386x | 24 | 24 |
| rf880_fusion_compression_host_staged | gpu_only | 24.749 | 1.000x | 0 | 0 |
| rf880_fusion_compression_host_staged | static_all_fpga | 19.026 | 1.301x | 49 | 73 |
| rf880_fusion_compression_host_staged | communication_eft | 16.872 | 1.467x | 24 | 24 |
| rf880_fusion_compression_shared_single_buffer | gpu_only | 24.749 | 1.000x | 0 | 0 |
| rf880_fusion_compression_shared_single_buffer | static_all_fpga | 17.231 | 1.436x | 49 | 73 |
| rf880_fusion_compression_shared_single_buffer | communication_eft | 16.228 | 1.525x | 36 | 48 |
| rf880_fusion_compression_shared_double_buffer | gpu_only | 24.749 | 1.000x | 0 | 0 |
| rf880_fusion_compression_shared_double_buffer | static_all_fpga | 17.231 | 1.436x | 49 | 73 |
| rf880_fusion_compression_shared_double_buffer | communication_eft | 16.228 | 1.525x | 36 | 48 |
| rf880_direct_dma_upper_bound | gpu_only | 24.749 | 1.000x | 0 | 0 |
| rf880_direct_dma_upper_bound | static_all_fpga | 17.231 | 1.436x | 49 | 73 |
| rf880_direct_dma_upper_bound | communication_eft | 16.228 | 1.525x | 36 | 48 |

## Best Results

- Best single-request communication-aware case: rf880_fusion_compression_shared_single_buffer, 16.228 ms, 1.525x vs its GPU-only baseline.
- Best 16-request throughput case: rf880_fusion_compression_shared_single_buffer / communication_eft, 162.923 req/s, 4.032x.
- Best design-space analytical condition: shared_mapped_one_copy, 7.2 GB/s, 10 us, FPGA speedup 16x.

## Dynamic Load

| Offered rate (req/s) | Policy | P95 latency (ms) | SLO violation | Throughput (req/s) |
| ---: | --- | ---: | ---: | ---: |
| 60 | adaptive_dual_mode | 20.228 | 0.000 | 60.000 |
| 60 | gpu_only | 1843.802 | 1.000 | 40.406 |
| 60 | latency_eft | 20.228 | 0.000 | 60.000 |
| 100 | adaptive_dual_mode | 26.108 | 0.000 | 100.000 |
| 100 | gpu_only | 1843.802 | 1.000 | 40.406 |
| 100 | latency_eft | 26.108 | 0.000 | 100.000 |
| 140 | adaptive_dual_mode | 42.933 | 1.000 | 140.000 |
| 140 | gpu_only | 1843.802 | 1.000 | 40.406 |
| 140 | latency_eft | 42.933 | 1.000 | 140.000 |

## Hardware Bring-Up Boundary

- Generated task-measurement rows: 138
- Generated transfer-measurement rows: 60
- Largest planned payload: 4194304 bytes

The next blocking step is target measurement: confirm PCIe LnkSta, run GPU kernels, run FPGA Linear/fused kernels, measure DMA paths, then rebuild a complete target profile with `simgrid_baseline/16_hardware_calibration/validate_and_build_profile.py`.
