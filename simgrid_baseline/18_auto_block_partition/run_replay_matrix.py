"""Run each SimGrid scenario in its own process and validate event traces."""

import csv
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "replay"
REQUEST_COUNTS = (1, 8, 16)


def replay_one(policy, requests, buffers):
    directory = OUTPUT / policy / f"requests_{requests}_buffers_{buffers}"
    command = [sys.executable, str(HERE / "partition.py"), "--output", str(directory),
               "--simgrid-policy", policy, "--simgrid-requests", str(requests),
               "--simgrid-buffers", str(buffers)]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"SimGrid replay failed: {directory}\n{completed.stderr}")
    summary = json.loads((directory / "simgrid_replay_summary.json").read_text(encoding="utf-8"))
    trace = json.loads((directory / "simgrid_trace.json").read_text(encoding="utf-8"))
    dag = json.loads((directory / "mixed_dag.json").read_text(encoding="utf-8"))
    original = json.loads((HERE.parent / "08_coarse_vit_dag" / "results" /
                           "vit_batch1_coarse_dag.json").read_text(encoding="utf-8"))
    task_count = len(original["tasks"]) if policy != "automatic_region_dp" else len(dag["tasks"])
    comparison = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["comparison"]
    expected_transfers = comparison[policy]["physical_transfer_stages"]
    if summary["task_count"] != task_count * requests:
        raise RuntimeError(f"Incomplete task trace: {directory}")
    if summary["transfer_stage_count"] != expected_transfers * requests:
        raise RuntimeError(f"Incomplete transfer trace: {directory}")
    if len(summary["request_completion_us"]) != requests:
        raise RuntimeError(f"Incomplete completion trace: {directory}")
    if len(trace["tasks"]) != summary["task_count"] or len(trace["transfers"]) != summary["transfer_stage_count"]:
        raise RuntimeError(f"Summary/trace count mismatch: {directory}")
    if any(row["finish_us"] < row["start_us"] for row in trace["tasks"] + trace["transfers"]):
        raise RuntimeError(f"Negative event interval: {directory}")
    if summary["execution_mode"] != "simgrid_event_replay_sensitivity_parameters" or summary["valid_for_target_prediction"]:
        raise RuntimeError(f"Incorrect evidence label: {directory}")
    return summary


def run_matrix():
    summaries = {}
    for requests in REQUEST_COUNTS:
        for policy, buffers in (("gpu_only", 1), ("static_all_linear_fpga", 1),
                                ("automatic_region_dp", 1), ("automatic_region_dp", 2)):
            summaries[(policy, requests, buffers)] = replay_one(policy, requests, buffers)
    rows = []
    for requests in REQUEST_COUNTS:
        gpu = summaries[("gpu_only", requests, 1)]
        one = summaries[("automatic_region_dp", requests, 1)]
        for policy, buffers in (("gpu_only", 1), ("static_all_linear_fpga", 1),
                                ("automatic_region_dp", 1), ("automatic_region_dp", 2)):
            summary = summaries[(policy, requests, buffers)]
            makespan = summary["makespan_us"]
            rows.append({
                "status": "simgrid_sensitivity_not_hardware_measurement",
                "policy": policy,
                "requests": requests,
                "buffers": buffers,
                "batch_makespan_ms": round(makespan / 1000.0, 6),
                "batch_throughput_requests_per_s": round(requests * 1_000_000.0 / makespan, 6),
                "batch_speedup_vs_gpu_only": round(gpu["makespan_us"] / makespan, 6),
                "batch_speedup_vs_one_buffer": round(one["makespan_us"] / makespan, 6)
                    if policy == "automatic_region_dp" else "",
                "task_events": summary["task_count"],
                "transfer_stage_events": summary["transfer_stage_count"],
            })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "matrix.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def main():
    rows = run_matrix()
    print(json.dumps({"output": str(OUTPUT / "matrix.csv"), "scenarios": len(rows),
                      "rows": [{key: row[key] for key in ("policy", "requests", "buffers",
                            "batch_makespan_ms", "batch_throughput_requests_per_s")}
                            for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
