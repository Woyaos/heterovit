# Experiment 14 Results

All values are sensitivity results and are invalid for target-hardware prediction.

## Communication-Aware Single-Request Ablation

| Case | Makespan (ms) | Speedup vs GPU | FPGA Linear equivalents | Cross edges | Encoded bytes (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_linear_only | 22.184 | 1.000x | 0 | 0 | 0.000 |
| compression_2x | 22.184 | 1.000x | 0 | 0 | 0.000 |
| compression_4x | 22.184 | 1.000x | 0 | 0 | 0.000 |
| ffn_fusion_residency | 18.910 | 1.173x | 24 | 24 | 14.524 |
| shared_mapped_one_copy | 22.184 | 1.000x | 0 | 0 | 0.000 |
| fusion_plus_compression | 17.299 | 1.282x | 24 | 24 | 7.262 |
| fusion_compression_one_copy_single_buffer | 16.798 | 1.321x | 24 | 24 | 7.262 |
| fusion_compression_one_copy_double_buffer | 16.798 | 1.321x | 24 | 24 | 7.262 |
| fusion_compression_direct_dma_double_buffer | 16.798 | 1.321x | 24 | 24 | 7.262 |

## Static-All Diagnostic

| Case | Makespan (ms) | Speedup vs GPU | Cross edges | Encoded bytes (MB) |
| --- | ---: | ---: | ---: | ---: |
| baseline_linear_only | 48.870 | 0.454x | 97 | 116.198 |
| compression_2x | 35.605 | 0.623x | 97 | 58.099 |
| compression_4x | 26.455 | 0.839x | 97 | 29.050 |
| ffn_fusion_residency | 29.784 | 0.745x | 73 | 58.101 |
| shared_mapped_one_copy | 42.285 | 0.525x | 97 | 116.198 |
| fusion_plus_compression | 23.249 | 0.954x | 73 | 29.050 |
| fusion_compression_one_copy_single_buffer | 21.359 | 1.039x | 73 | 29.050 |
| fusion_compression_one_copy_double_buffer | 21.359 | 1.039x | 73 | 29.050 |
| fusion_compression_direct_dma_double_buffer | 21.359 | 1.039x | 73 | 29.050 |

## Communication-Aware Eight-Request Pipeline

| Case | Throughput (req/s) | Throughput speedup | Input buffers |
| --- | ---: | ---: | ---: |
| baseline_linear_only | 45.077 | 1.000x | 1 |
| compression_2x | 45.077 | 1.000x | 1 |
| compression_4x | 45.077 | 1.000x | 1 |
| ffn_fusion_residency | 100.070 | 2.220x | 1 |
| shared_mapped_one_copy | 45.077 | 1.000x | 1 |
| fusion_plus_compression | 100.985 | 2.240x | 1 |
| fusion_compression_one_copy_single_buffer | 101.065 | 2.242x | 1 |
| fusion_compression_one_copy_double_buffer | 102.227 | 2.268x | 2 |
| fusion_compression_direct_dma_double_buffer | 102.227 | 2.268x | 2 |

## Buffer and Objective Comparison

| Case | Policy | Throughput (req/s) | Speedup vs GPU |
| --- | --- | ---: | ---: |
| fusion_compression_one_copy_single_buffer | static_all_fpga | 73.742 | 1.636x |
| fusion_compression_one_copy_single_buffer | communication_eft | 101.065 | 2.242x |
| fusion_compression_one_copy_double_buffer | static_all_fpga | 105.952 | 2.350x |
| fusion_compression_one_copy_double_buffer | communication_eft | 102.227 | 2.268x |
