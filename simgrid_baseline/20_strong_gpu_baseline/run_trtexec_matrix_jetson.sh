#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-vit.onnx}
OUT=${2:-gpu_baseline_raw}
mkdir -p "$OUT"

for AUX in 0 1 2 4; do
  ENGINE="$OUT/vit_fp16_aux${AUX}.engine"
  trtexec --onnx="$MODEL" --fp16 \
    --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:1x3x224x224 \
    --builderOptimizationLevel=5 --maxAuxStreams="$AUX" \
    --timingCacheFile="$OUT/vit.timing.cache" --profilingVerbosity=detailed \
    --saveEngine="$ENGINE" --skipInference \
    > "$OUT/build_aux${AUX}.log" 2>&1

  for GRAPH in off on; do
    for TRANSFERS in off on; do
      EXTRA=()
      [[ "$GRAPH" == on ]] && EXTRA+=(--useCudaGraph)
      [[ "$TRANSFERS" == on ]] && EXTRA+=(--includeDataTransfers)
      trtexec --loadEngine="$ENGINE" --shapes=input:1x3x224x224 \
        --warmUp=1000 --duration=30 --iterations=200 --dumpProfile \
        "${EXTRA[@]}" \
        > "$OUT/run_aux${AUX}_graph${GRAPH}_transfers${TRANSFERS}.log" 2>&1
    done
  done
done
