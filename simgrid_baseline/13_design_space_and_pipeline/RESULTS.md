# Experiment 13 Results

## Coverage

- 3 communication-path abstractions.
- 4 effective PCIe bandwidths: 2, 4, 8, and 16 GB/s.
- 4 one-way fixed latencies: 5, 10, 20, and 50 us.
- 4 FPGA Linear compute speedups: 2x, 4x, 8x, and 16x.
- 192 hardware conditions and 576 policy rows.
- 4 representative scenarios replayed in SimGrid with 1, 2, 4, and 8
  simultaneous requests, producing 48 replay rows.

All results are sensitivity results and are invalid for target prediction.

## Design-Space Result

| Path abstraction | Profitable EFT conditions | Profitable static conditions | Maximum conditional EFT speedup |
| --- | ---: | ---: | ---: |
| Host-staged two-copy | 6 / 64 | 4 / 64 | 1.105x |
| Shared/mapped one-copy | 19 / 64 | 15 / 64 | 1.541x |
| Direct-DMA one-copy | 19 / 64 | 15 / 64 | 1.541x |

For the host-staged path, no tested point below 16 GB/s was profitable. At
16 GB/s, profitable points required at least 8x FPGA Linear speedup and at most
20 us one-way fixed PCIe latency. No host-staged point at 50 us was profitable.

The best tested analytical point used a one-copy path, 16 GB/s, 5 us, and 16x
FPGA Linear speedup. Communication-aware placement offloaded 48 Linears and
reached a conditional 1.541x speedup over GPU-only.

Shared/mapped and direct-DMA paths intentionally have identical results when
given identical effective bandwidth and latency. The model distinguishes their
data routes, but hardware-specific synchronization and driver overhead are not
available yet.

## SimGrid Pipeline Result

Selected one-request and eight-request results:

| Scenario and policy | 1-request speedup | 8-request throughput speedup | FPGA Linears |
| --- | ---: | ---: | ---: |
| Staged reference, static all | 0.454x | 0.723x | 49 |
| Staged reference, communication EFT | 1.000x | 1.000x | 0 |
| Staged favorable, static all | 0.990x | 2.701x | 49 |
| Staged favorable, communication EFT | 1.026x | 2.882x | 36 |
| One-copy midpoint, static all | 0.851x | 1.435x | 49 |
| One-copy midpoint, communication EFT | 1.000x | 1.000x | 0 |

The favorable staged communication-aware case completed eight requests in
61.577 ms, or 129.918 requests/s, versus GPU-only at 45.077 requests/s. It
executed 1,000 task records and 1,152 transfer-stage records, exactly matching
the expected DAG and two-stage communication counts.

The one-copy midpoint exposes an objective conflict: static offload hurts
single-request latency but improves saturated throughput. The current EFT
policy optimizes single-request completion and therefore keeps GPU-only at this
point. A real runtime must expose separate latency and throughput modes rather
than report throughput gain as latency gain.

## Validation

`validation_report.json` passes all 12 checks, including row counts, unique
keys, complete policy coverage, task and transfer records, analytical EFT
safety, one-copy equivalence under matched parameters, and target-prediction
guards.

## Hardware-Free Conclusion

The simulator, baselines, path abstractions, latency policy, pipeline replay,
and calibration interfaces are complete. Further numerical refinement without
hardware would add assumptions rather than evidence. The next valid numerical
step is target microbenchmark calibration.
