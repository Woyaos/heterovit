#!/usr/bin/env python3
"""Validate that the heterogeneous system is compared with a strong GPU baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


AUX_STREAMS = (0, 1, 2, 4)
REQUIRED_KEYS = {
    (aux, graph, transfers)
    for aux in AUX_STREAMS
    for graph in (False, True)
    for transfers in (False, True)
}
NUMERIC_FIELDS = (
    "median_ms",
    "p95_ms",
    "mean_ms",
    "throughput_requests_per_s",
    "enqueue_ms",
)


def as_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean: {value}")


def validate(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    measured = []
    errors = []
    keys = set()
    for index, row in enumerate(rows, start=2):
        if not row.get("median_ms", "").strip():
            continue
        try:
            key = (int(row["aux_streams"]), as_bool(row["cuda_graph"]), as_bool(row["include_transfers"]))
            if key in keys:
                errors.append(f"row {index}: duplicate configuration {key}")
            keys.add(key)
            if row["precision"] != "FP16" or int(row["batch"]) != 1:
                errors.append(f"row {index}: expected FP16 batch 1")
            for field in NUMERIC_FIELDS:
                if float(row[field]) < 0:
                    errors.append(f"row {index}: {field} must be non-negative")
            if int(row["samples"]) < 200:
                errors.append(f"row {index}: at least 200 measured samples required")
            for field in ("model_sha256", "engine_sha256", "jetson_sku", "power_mode", "trt_version"):
                if not row[field].strip():
                    errors.append(f"row {index}: missing {field}")
            measured.append(row)
        except (KeyError, ValueError) as exc:
            errors.append(f"row {index}: {exc}")

    missing = sorted(REQUIRED_KEYS - keys)
    ready = not errors and not missing
    compute_rows = [row for row in measured if not as_bool(row["include_transfers"])]
    e2e_rows = [row for row in measured if as_bool(row["include_transfers"])]
    best_compute = min(compute_rows, key=lambda row: float(row["median_ms"])) if compute_rows else None
    best_e2e = min(e2e_rows, key=lambda row: float(row["median_ms"])) if e2e_rows else None
    return {
        "status": "ready" if ready else "waiting_for_jetson_measurements",
        "valid_for_gpu_baseline_claim": ready,
        "measured_rows": len(measured),
        "required_rows": len(REQUIRED_KEYS),
        "missing_configurations": [
            {"aux_streams": aux, "cuda_graph": graph, "include_transfers": transfers}
            for aux, graph, transfers in missing
        ],
        "errors": errors,
        "best_compute_only": best_compute,
        "best_end_to_end": best_e2e,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "measurements",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("gpu_baseline_measurements.csv"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.measurements)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if result["valid_for_gpu_baseline_claim"] else 2)


if __name__ == "__main__":
    main()
