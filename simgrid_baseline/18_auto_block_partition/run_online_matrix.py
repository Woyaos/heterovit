"""SimGrid online-arrival comparison for the fixed DP map and GPU-only."""

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import partition


HERE = Path(__file__).resolve().parent
ONLINE = HERE / "results" / "online"
DYNAMIC = HERE.parent / "15_dynamic_runtime" / "run_dynamic_case.py"
REQUESTS = 64
RATES = (40, 60, 100)
SLO_US = 30000.0


def make_profiles():
    base = json.loads(partition.PROFILE_PATH.read_text(encoding="utf-8"))
    paths = {}
    directory = ONLINE / "profiles"
    directory.mkdir(parents=True, exist_ok=True)
    for buffers in (1, 2):
        profile = copy.deepcopy(base)
        profile["pipeline"]["fpga_input_buffers"] = buffers
        path = directory / f"rf880_assumed_buffers_{buffers}.json"
        path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        paths[buffers] = path
    return paths


def run_one(policy, buffers, rate, profile):
    directory = ONLINE / f"rate_{rate}" / f"{policy}_buffers_{buffers}"
    dag = HERE / "results" / "default" / "mixed_dag.json"
    command = [sys.executable, str(DYNAMIC), policy, str(dag), str(profile),
               str(directory), "--requests", str(REQUESTS), "--arrival-rate", str(rate),
               "--slo-us", str(SLO_US)]
    if policy == "fixed_manifest":
        command.extend(["--placements", str(HERE / "results" / "default" / "placements.json")])
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"Online SimGrid failed: {directory}\n{completed.stderr}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "request_completion.csv").open(newline="", encoding="utf-8") as handle:
        completions = list(csv.DictReader(handle))
    task_count = len(json.loads(dag.read_text(encoding="utf-8"))["tasks"])
    expected_transfers = 0 if policy == "gpu_only" else 48
    if len(completions) != REQUESTS or summary["task_records"] != task_count * REQUESTS:
        raise RuntimeError(f"Incomplete online task/completion trace: {directory}")
    if summary["transfer_records"] != expected_transfers * REQUESTS:
        raise RuntimeError(f"Incomplete online transfer trace: {directory}")
    if summary["valid_for_target_prediction"] or summary["policy"] != policy:
        raise RuntimeError(f"Incorrect online evidence label: {directory}")
    for row in completions:
        arrival = float(row["arrival_us"])
        completion = float(row["completion_us"])
        latency = float(row["latency_us"])
        if abs((completion - arrival) - latency) > 1e-5:
            raise RuntimeError(f"Incorrect per-request latency: {directory}")
    return summary


def run_matrix():
    profiles = make_profiles()
    rows = []
    for rate in RATES:
        for policy, buffers in (("gpu_only", 1), ("fixed_manifest", 1),
                                ("fixed_manifest", 2)):
            summary = run_one(policy, buffers, rate, profiles[buffers])
            rows.append({
                "status": "simgrid_online_sensitivity_not_hardware_measurement",
                "policy": policy,
                "offered_rate_requests_per_s": rate,
                "requests": REQUESTS,
                "buffers": buffers,
                "mean_latency_ms": round(summary["mean_latency_us"] / 1000.0, 6),
                "p95_latency_ms": round(summary["p95_latency_us"] / 1000.0, 6),
                "slo_violation_rate": round(summary["slo_violation_rate"], 6),
                "achieved_throughput_requests_per_s": round(
                    summary["achieved_throughput_requests_per_s"], 6),
                "task_events": summary["task_records"],
                "transfer_stage_events": summary["transfer_records"],
            })
    with (ONLINE / "online_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ONLINE / "online_matrix.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def main():
    rows = run_matrix()
    print(json.dumps({"output": str(ONLINE / "online_matrix.csv"), "scenarios": len(rows),
                      "rows": [{key: row[key] for key in ("policy", "offered_rate_requests_per_s",
                            "buffers", "p95_latency_ms", "achieved_throughput_requests_per_s")}
                            for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
