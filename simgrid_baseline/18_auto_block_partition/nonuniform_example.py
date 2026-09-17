"""Reproducible synthetic example showing layer-selective FFN placement."""

import json
from pathlib import Path

import partition


def main():
    runtime = partition.load_runtime()
    dag = json.loads(partition.DAG_PATH.read_text(encoding="utf-8"))
    profile = json.loads(partition.PROFILE_PATH.read_text(encoding="utf-8"))
    capability = json.loads(partition.CAPABILITY_PATH.read_text(encoding="utf-8"))
    regions, _ = partition.ffns_from_dependencies(dag)
    profile["task_costs_us"] = {
        task_id: {"GPU": 100.0} for task_id in regions[0]["task_ids"]
    }
    profile["profile_name"] += "_synthetic_nonuniform_example"
    profile["description"] += " Synthetic first-FFN GPU costs; not board measurements."
    result = partition.run(dag, profile, capability, runtime)
    output = Path(__file__).with_name("results") / "nonuniform_example.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "status": "synthetic_not_hardware_measurement",
        "first_ffn_gpu_task_cost_us_assumed": 100.0,
        "first_ffn_task_ids": regions[0]["task_ids"],
        "selected_fpga_ffn_ids": [
            solution["id"] for solution in result["regions"]
            if len(solution["blocks"]) == 1 and solution["blocks"][0]["device"] == "FPGA"
        ],
        "comparison": result["comparison"],
    }
    output.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "selected_fpga_ffns": len(content["selected_fpga_ffn_ids"]),
                      "automatic_dp_us": content["comparison"]["automatic_region_dp"]["makespan_us"],
                      "fixed_ffn_us": content["comparison"]["fixed_ffn"]["makespan_us"]}, indent=2))


if __name__ == "__main__":
    main()
