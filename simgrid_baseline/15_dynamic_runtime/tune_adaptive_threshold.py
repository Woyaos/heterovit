import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


RESULT_PREFIX = "RESULT_JSON "


def parse_result(completed, label):
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(f"Missing result for {label}:\n{completed.stdout}\n{completed.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().parent / "run_dynamic_case.py"
    rows = []
    for rate in config["threshold_tuning_rates_requests_per_s"]:
        for threshold in config["adaptive_threshold_candidates"]:
            case_output = output / f"rate_{rate}" / f"threshold_{threshold}"
            command = [
                sys.executable,
                str(runner),
                "adaptive_dual_mode",
                str(args.dag.resolve()),
                str(args.profile.resolve()),
                str(case_output),
                "--requests",
                str(config["request_count"]),
                "--arrival-rate",
                str(rate),
                "--adaptive-threshold",
                str(threshold),
                "--slo-us",
                str(config["slo_us"]),
            ]
            result = parse_result(
                subprocess.run(command, text=True, capture_output=True, check=True),
                f"{rate}/{threshold}",
            )
            rows.append(
                {
                    "offered_rate_requests_per_s": rate,
                    "backlog_threshold": threshold,
                    "mean_latency_us": result["mean_latency_us"],
                    "p95_latency_us": result["p95_latency_us"],
                    "slo_violation_rate": result["slo_violation_rate"],
                    "achieved_throughput_requests_per_s": result[
                        "achieved_throughput_requests_per_s"
                    ],
                    "latency_mode_requests": result["mode_counts"].get("latency_eft", 0),
                    "throughput_mode_requests": result["mode_counts"].get("throughput_all", 0),
                    "valid_for_target_prediction": False,
                }
            )

    with (output / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best_by_rate = {}
    for rate in config["threshold_tuning_rates_requests_per_s"]:
        candidates = [row for row in rows if row["offered_rate_requests_per_s"] == rate]
        best_by_rate[str(rate)] = min(
            candidates,
            key=lambda row: (
                row["slo_violation_rate"],
                row["p95_latency_us"],
                -row["achieved_throughput_requests_per_s"],
            ),
        )
    summary = {
        "valid_for_target_prediction": False,
        "selection_objective": "minimum SLO violation, then P95 latency, then maximum throughput",
        "best_by_rate": best_by_rate,
        "rows": len(rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
