# Experiment 19: Single-Inference Tile Streaming

This experiment evaluates token-tiled execution inside one fused ViT FFN. It is
separate from the existing multi-request pipeline: the optimization target is
the latency of one inference.

It compares:

- whole-tensor DMA plus sequential FC1/GELU/FC2;
- whole-tensor DMA plus an internal three-process dataflow pipeline;
- end-to-end tile streaming with one or two input/output buffer banks;
- one shared FC engine versus concurrent FC1 and FC2 engines;
- one streaming DMA descriptor versus a separate DMA submission per tile.

Run:

```powershell
python simgrid_baseline/19_single_inference_tile_streaming/tile_streaming_model.py
python -m unittest discover simgrid_baseline/19_single_inference_tile_streaming -p "test*.py"
```

The model uses the existing FP16 RF880-AGX sensitivity profile and reproduces
the existing 638.29952 us full-FFN interval before exploring tiles. Every output
is marked `valid_for_target_prediction=false`.

## Interpretation

Token tiling is semantically valid for the FFN because the same two Linear
layers and GELU are applied independently to each token. Attention is not
modeled as token-independent and remains a synchronization barrier between
Transformer blocks.

An internal pipeline requires independent FC1, GELU, and FC2 hardware processes,
or enough replicated resources to sustain the modeled overlap. A shared matrix
engine can reduce storage through tiling but cannot obtain the same latency
overlap. End-to-end streaming additionally requires a DMA/driver path that can
produce and consume chunks under one amortized transfer setup. Submitting one
DMA operation per small tile can be slower because every tile pays fixed PCIe
and driver latency.

The conservative memory estimate includes ping-pong copies of the input, both
expanded intermediate arrays, and output, plus both low-bit weight matrices.
The default uses two bits per ternary weight because `-1/0/+1` has three states.
FIFO streaming can use less memory, but that must be established by HLS/RTL
co-simulation rather than assumed here.
