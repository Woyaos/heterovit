import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


POLICIES = (
    "gpu_only",
    "static_all_fpga",
    "iris_profile_compute_only",
    "iris_data_locality",
    "communication_eft",
)
RESULT_PREFIX = "RESULT_JSON "


def run_policy(script, policy, dag, profile, output_directory):
    command = [
        sys.executable,
        str(script),
        policy,
        str(dag),
        str(profile),
        str(output_directory / policy),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(f"No result marker for {policy}:\n{completed.stdout}\n{completed.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / "run_policy.py"
    results = [
        run_policy(script, policy, args.dag.resolve(), args.profile.resolve(), args.output_directory.resolve())
        for policy in POLICIES
    ]
    gpu_only = next(row for row in results if row["policy"] == "gpu_only")
    baseline_us = gpu_only["simgrid_makespan_us"]
    comparison_rows = []
    for result in results:
        comparison_rows.append(
            {
                "policy": result["policy"],
                "simgrid_makespan_us": result["simgrid_makespan_us"],
                "speedup_vs_gpu_only": baseline_us / result["simgrid_makespan_us"],
                "offloaded_linear_tasks": result["offloaded_linear_tasks"],
                "cross_device_edges": result["cross_device_edges"],
                "logical_cross_device_bytes": result["logical_cross_device_bytes"],
                "host_staged_link_bytes": result["host_staged_link_bytes"],
                "valid_for_target_prediction": result["valid_for_target_prediction"],
            }
        )

    csv_path = args.output_directory / "policy_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {
        "profile_kind": results[0]["profile_kind"],
        "profile_name": results[0]["profile_name"],
        "valid_for_target_prediction": False,
        "policies": comparison_rows,
        "best_policy_under_sensitivity_profile": min(
            comparison_rows, key=lambda row: row["simgrid_makespan_us"]
        )["policy"],
        "comparison_csv": str(csv_path),
    }
    (args.output_directory / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Policy                         Time (us)  Speedup  FPGA Linear  Cross edges  Logical MB")
    print("-" * 92)
    for row in comparison_rows:
        print(
            f"{row['policy']:<30} {row['simgrid_makespan_us']:>10.3f}  "
            f"{row['speedup_vs_gpu_only']:>7.3f}  {row['offloaded_linear_tasks']:>11}  "
            f"{row['cross_device_edges']:>11}  {row['logical_cross_device_bytes'] / 1_000_000:>10.3f}"
        )
    print("\nAll values are sensitivity results, not target-hardware predictions.")


if __name__ == "__main__":
    main()
