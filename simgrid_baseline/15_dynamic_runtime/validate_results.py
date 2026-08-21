import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


POLICIES = {"gpu_only", "latency_eft", "throughput_all", "adaptive_dual_mode"}


def as_float(row, key):
    return float(row[key])


def check(condition, message, failures):
    if not condition:
        failures.append(message)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            f"Usage: python {sys.argv[0]} <dynamic-config.json> <load-sweep.csv> <threshold-sweep.csv>"
        )

    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = list(csv.DictReader(Path(sys.argv[2]).open(encoding="utf-8")))
    threshold_rows = list(csv.DictReader(Path(sys.argv[3]).open(encoding="utf-8")))
    failures = []
    expected_cases = len(config["arrival_rates_requests_per_s"]) * len(config["policies"])
    check(len(rows) == expected_cases, f"expected {expected_cases} load cases, got {len(rows)}", failures)

    by_rate = defaultdict(dict)
    for row in rows:
        rate = int(row["offered_rate_requests_per_s"])
        by_rate[rate][row["policy"]] = row
        check(row["valid_for_target_prediction"].lower() == "false", "target-valid flag must be false", failures)
        check(int(row["request_count"]) == config["request_count"], "request count mismatch", failures)
        check(int(row["task_records"]) > 0, "empty task timeline", failures)

    for rate in config["arrival_rates_requests_per_s"]:
        check(set(by_rate[rate]) == POLICIES, f"rate {rate} is missing policies", failures)

    # Structural conclusions only. These are sensitivity results, not target-time claims.
    if 20 in by_rate and set(by_rate[20]) == POLICIES:
        check(
            as_float(by_rate[20]["latency_eft"], "p95_latency_us")
            < as_float(by_rate[20]["gpu_only"], "p95_latency_us"),
            "latency EFT should beat GPU-only P95 at low load",
            failures,
        )
    if 60 in by_rate and set(by_rate[60]) == POLICIES:
        check(
            as_float(by_rate[60]["gpu_only"], "slo_violation_rate") > 0,
            "GPU-only should saturate by the configured 60 req/s sensitivity point",
            failures,
        )
        check(
            as_float(by_rate[60]["latency_eft"], "slo_violation_rate") == 0,
            "latency EFT should still meet the configured SLO at 60 req/s",
            failures,
        )
    highest_rate = max(config["arrival_rates_requests_per_s"])
    if highest_rate in by_rate and "adaptive_dual_mode" in by_rate[highest_rate]:
        adaptive = by_rate[highest_rate]["adaptive_dual_mode"]
        check(int(adaptive["latency_mode_requests"]) > 0, "adaptive policy never used latency mode", failures)
        check(int(adaptive["throughput_mode_requests"]) > 0, "adaptive policy never used throughput mode", failures)

    expected_threshold_cases = len(config["threshold_tuning_rates_requests_per_s"]) * len(
        config["adaptive_threshold_candidates"]
    )
    check(
        len(threshold_rows) == expected_threshold_cases,
        f"expected {expected_threshold_cases} threshold cases, got {len(threshold_rows)}",
        failures,
    )

    report = {
        "valid": not failures,
        "scope": "sensitivity-only structural validation",
        "load_cases": len(rows),
        "threshold_cases": len(threshold_rows),
        "failures": failures,
    }
    report_path = Path(sys.argv[2]).resolve().parent / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report": str(report_path)}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
