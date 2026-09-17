"""Sensitivity sweep for the FFN block partition under unmeasured RF880 parameters."""

import argparse
import copy
import csv
import json
from pathlib import Path

import partition


def run_sweep(dag, profile, capability, runtime):
    rows = []
    for speedup in (1.0, 2.0, 4.0, 8.0):
        for pcie_fixed_us in (0.0, 25.0, 100.0, 300.0, 600.0):
            for pcie_gbps in (1.5, 3.2, 6.4):
                scenario = copy.deepcopy(profile)
                scenario["fpga"]["linear_speedup_over_gpu"] = speedup
                scenario["communication_path"]["host_fpga_pcie"]["fixed_latency_us"] = pcie_fixed_us
                scenario["communication_path"]["host_fpga_pcie"]["effective_bandwidth_GBps"] = pcie_gbps
                result = partition.run(dag, scenario, capability, runtime)
                comparison = result["comparison"]
                gpu_us = comparison["gpu_only"]["makespan_us"]
                chosen_us = comparison["automatic_region_dp"]["makespan_us"]
                full_ffns = sum(
                    len(solution["blocks"]) == 1
                    and solution["blocks"][0]["device"] == "FPGA"
                    and len(solution["blocks"][0]["task_ids"]) == 3
                    for solution in result["regions"]
                )
                singleton_linears = sum(
                    block["device"] == "FPGA" and len(block["task_ids"]) == 1
                    for solution in result["regions"] for block in solution["blocks"]
                )
                rows.append({
                    "linear_speedup_assumed": speedup,
                    "pcie_fixed_us_assumed": pcie_fixed_us,
                    "pcie_bandwidth_GBps_assumed": pcie_gbps,
                    "selected_full_ffns": full_ffns,
                    "selected_singleton_linears": singleton_linears,
                    "gpu_only_us": round(gpu_us, 6),
                    "fixed_ffn_us": round(comparison["fixed_ffn"]["makespan_us"], 6),
                    "automatic_dp_us": round(chosen_us, 6),
                    "automatic_vs_gpu_speedup": round(gpu_us / chosen_us, 6),
                    "automatic_vs_fixed_ffn_speedup": round(
                        comparison["fixed_ffn"]["makespan_us"] / chosen_us, 6),
                    "cross_device_edges": comparison["automatic_region_dp"]["cross_device_edges"],
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("results") / "sensitivity.csv")
    args = parser.parse_args()
    runtime = partition.load_runtime()
    dag = json.loads(partition.DAG_PATH.read_text(encoding="utf-8"))
    profile = json.loads(partition.PROFILE_PATH.read_text(encoding="utf-8"))
    capability = json.loads(partition.CAPABILITY_PATH.read_text(encoding="utf-8"))
    rows = run_sweep(dag, profile, capability, runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "scenarios": len(rows),
                      "gpu_wins_or_ties": sum(row["automatic_vs_gpu_speedup"] <= 1 for row in rows),
                      "partitions_differ_from_fixed_ffn": sum(
                          row["selected_full_ffns"] < 12 or row["selected_singleton_linears"] > 0
                          for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
