# Experiment 12 Results

## Scope

The complete 125-task, 148-edge ViT DAG was replayed in SimGrid. Every
cross-device edge used two explicit communication stages through HOST. These
results validate scheduler behavior under declared sensitivity assumptions;
they are not target-hardware predictions.

## Reference Host-Staged FP32 Point

| Policy | Makespan (us) | Speedup vs GPU | FPGA Linears | Cross edges |
| --- | ---: | ---: | ---: | ---: |
| GPU-only | 22,184.376 | 1.000 | 0 | 0 |
| Static all-Linear FPGA | 48,870.450 | 0.454 | 49 | 97 |
| IRIS-profile-like | 48,840.809 | 0.454 | 48 | 96 |
| IRIS-data-like | 22,184.376 | 1.000 | 0 | 0 |
| Communication-aware EFT | 22,184.376 | 1.000 | 0 | 0 |

Under this conditional point, compute-only placement is misled by faster FPGA
kernels and more than doubles the makespan. Communication-aware placement
rejects every offload after including the mandatory return to a GPU successor.

## Favorable Host-Staged FP32 Point

| Policy | Makespan (us) | Speedup vs GPU | FPGA Linears | Cross edges |
| --- | ---: | ---: | ---: | ---: |
| GPU-only | 22,184.376 | 1.000 | 0 | 0 |
| Static all-Linear FPGA | 18,830.446 | 1.178 | 49 | 97 |
| IRIS-profile-like | 18,830.446 | 1.178 | 49 | 97 |
| IRIS-data-like | 22,184.376 | 1.000 | 0 | 0 |
| Communication-aware EFT | 18,559.331 | 1.195 | 36 | 72 |

The communication-aware policy offloads all 12 QKV, all 12 FFN1, and all 12
FFN2 tasks. It keeps all attention projections and the classifier on GPU. This
selective placement beats both GPU-only and static all-Linear offload under the
declared favorable assumptions.

## Structural Validation

- Every policy executed exactly 125 task records.
- Every cross-device edge generated exactly two transfer-stage records.
- Every summary declares `valid_for_target_prediction=false`.
- GPU-only SimGrid and analytical makespans agree to floating-point precision.
- The communication-aware policy was checked in both reject-all and
  selectively-profitable offload conditions.

## Limitations

- All compute and link values are sensitivity assumptions.
- One inference request is modeled; no cross-request pipeline or throughput
  result is included.
- The GPU cost model is roofline-like rather than TensorRT measurement.
- FPGA costs are expressed as conditional speedup plus dispatch overhead.
- SimGrid CM02 transfer timing differs slightly from the analytical selector,
  so SimGrid replay is the reported makespan.
- Pinned memory, mapped memory, direct DMA, cache synchronization, layout
  conversion, and power are not modeled yet.
