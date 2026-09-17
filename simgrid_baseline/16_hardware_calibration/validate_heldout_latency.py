"""Compare calibrated SimGrid predictions with separate board measurements."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    "policy", "sample_id", "latency_us", "source_kind", "source_ref",
    "dataset_role", "model_variant", "activation_precision",
}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def compare(predictions, rows, minimum_samples=30):
    profiles = {item.get("profile_name") for item in predictions.values()}
    if None in profiles or len(profiles) != 1:
        raise ValueError("Predictions must use one named calibrated target profile")
    grouped = defaultdict(list)
    provenance = defaultdict(set)
    sample_ids = defaultdict(set)
    for row in rows:
        policy = row["policy"]
        if policy not in predictions:
            raise ValueError(f"Unknown measured policy {policy}")
        prediction = predictions[policy]
        if row["source_kind"] != "measured" or row["dataset_role"] != "held_out_validation":
            raise ValueError(f"{policy} is not a held-out hardware measurement")
        if not row["source_ref"]:
            raise ValueError(f"{policy} has no measurement provenance")
        if row["model_variant"] != prediction["model_variant"]:
            raise ValueError(f"{policy} model variant differs from prediction")
        if row["activation_precision"] != prediction["activation_precision"]:
            raise ValueError(f"{policy} activation precision differs from prediction")
        if not row["sample_id"] or row["sample_id"] in sample_ids[policy]:
            raise ValueError(f"{policy} has a missing or duplicate sample ID")
        sample_ids[policy].add(row["sample_id"])
        try:
            latency = float(row["latency_us"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{policy} has an invalid latency") from error
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"{policy} latency must be positive and finite")
        grouped[policy].append(latency)
        provenance[policy].add(row["source_ref"])

    result = {}
    for policy, prediction in predictions.items():
        if (prediction.get("valid_for_target_prediction") is not True or
                prediction.get("execution_mode") !=
                "simgrid_event_replay_target_measurement_profile"):
            raise ValueError(f"{policy} prediction is not calibrated target replay")
        if prediction.get("requests") != 1:
            raise ValueError(f"{policy} prediction is not single-request replay")
        samples = grouped[policy]
        if len(samples) < minimum_samples:
            raise ValueError(f"{policy} needs at least {minimum_samples} held-out samples")
        median = statistics.median(samples)
        p95 = percentile(samples, 0.95)
        predicted = float(prediction["makespan_us"])
        if not math.isfinite(predicted) or predicted <= 0:
            raise ValueError(f"{policy} has invalid predicted latency")
        result[policy] = {
            "samples": len(samples),
            "predicted_us": predicted,
            "hardware_median_us": median,
            "hardware_p95_us": p95,
            "signed_median_error_us": predicted - median,
            "median_absolute_percentage_error": abs(predicted - median) / median,
            "hardware_source_refs": sorted(provenance[policy]),
        }
    return {"status": "held_out_target_board_validation",
            "minimum_samples_per_policy": minimum_samples,
            "mean_policy_median_absolute_percentage_error": statistics.mean(
                item["median_absolute_percentage_error"] for item in result.values()),
            "policies": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prediction", action="append", required=True,
                        help="POLICY=simgrid_replay_summary.json; repeat per policy")
    args = parser.parse_args()
    predictions = {}
    for item in args.prediction:
        policy, separator, file_path = item.partition("=")
        if not separator or not policy or not file_path or policy in predictions:
            parser.error("Each --prediction must be unique POLICY=path")
        prediction = json.loads(Path(file_path).read_text(encoding="utf-8"))
        if prediction["policy"] != policy:
            parser.error(f"Prediction file policy differs from {policy}")
        predictions[policy] = prediction
    with args.measurements.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
            parser.error("Held-out measurement CSV is missing required columns")
        rows = list(reader)
    result = compare(predictions, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "policies": len(result["policies"])}, indent=2))


if __name__ == "__main__":
    main()
