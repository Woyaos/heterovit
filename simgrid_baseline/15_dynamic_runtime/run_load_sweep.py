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


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("valid_for_target_prediction") is not False:
        raise RuntimeError("Dynamic config must declare valid_for_target_prediction=false")

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().parent / "run_dynamic_case.py"
    rows = []
    for rate in config["arrival_rates_requests_per_s"]:
        for policy in config["policies"]:
            case_output = output / f"rate_{rate}" / policy
            command = [
                sys.executable,
                str(runner),
                policy,
                str(args.dag.resolve()),
                str(args.profile.resolve()),
                str(case_output),
                "--requests",
                str(config["request_count"]),
                "--arrival-rate",
                str(rate),
                "--adaptive-threshold",
                str(config["adaptive_backlog_threshold"]),
                "--slo-us",
                str(config["slo_us"]),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=True)
            result = parse_result(completed, f"{rate}/{policy}")
            if result["task_records"] <= 0 or result["request_count"] != config["request_count"]:
                raise RuntimeError(f"Incomplete replay for {rate}/{policy}")
            rows.append(
                {
                    "offered_rate_requests_per_s": rate,
                    "policy": policy,
                    "request_count": result["request_count"],
                    "mean_latency_us": result["mean_latency_us"],
                    "p50_latency_us": result["p50_latency_us"],
                    "p95_latency_us": result["p95_latency_us"],
                    "max_latency_us": result["max_latency_us"],
                    "slo_violation_rate": result["slo_violation_rate"],
                    "achieved_throughput_requests_per_s": result[
                        "achieved_throughput_requests_per_s"
                    ],
                    "latency_mode_requests": result["mode_counts"].get("latency_eft", 0),
                    "throughput_mode_requests": result["mode_counts"].get("throughput_all", 0),
                    "gpu_only_requests": result["mode_counts"].get("gpu_only", 0),
                    "task_records": result["task_records"],
                    "transfer_records": result["transfer_records"],
                    "valid_for_target_prediction": False,
                }
            )

    write_csv(output / "dynamic_load_sweep.csv", rows)
    adaptive = [row for row in rows if row["policy"] == "adaptive_dual_mode"]
    summary = {
        "profile_kind": config["profile_kind"],
        "valid_for_target_prediction": False,
        "rows": len(rows),
        "request_count_per_case": config["request_count"],
        "adaptive_backlog_threshold": config["adaptive_backlog_threshold"],
        "slo_us": config["slo_us"],
        "adaptive_results": adaptive,
        "artifacts": {"sweep": "dynamic_load_sweep.csv", "cases": "rate_*/"},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
