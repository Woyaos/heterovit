# Single-Inference Tile Streaming Sensitivity Result

- Status: unmeasured sensitivity analysis
- Valid for RF880 target prediction: false
- Whole-FFN baseline: 638.300 us
- Recommended first HLS point: 282.370 us
- Conditional FFN speedup at that point: 2.261x
- Tile size: 16 tokens
- Buffers: 2
- DMA model: streaming_descriptor
- Compute architecture: independent_fc_engines
- Conservative activation buffers: 480.0 KiB
- Resident low-bit weights (2 bit): 1152.0 KiB
- Conditional full-model latency: 13.579 ms

The 16-token point is an initial synthesis candidate, not a proven optimum. The
conditional full-model value replaces each of the twelve existing monolithic
FPGA FFN intervals with that tile-stream result. It is not an RF880 prediction.
Per-tile DMA rows deliberately pay fixed startup for every tile; they show when
fine tiles lose to whole-tensor transfers. Hardware acceptance requires HLS/RTL
reports and measured DMA overlap.
