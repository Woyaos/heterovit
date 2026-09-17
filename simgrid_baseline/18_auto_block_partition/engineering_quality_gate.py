"""Industrial-style evidence gate for the RF880-AGX hetero-ViT study.

The gate does not prove target-board speedup. It checks whether the current
software evidence is internally consistent and whether it is labeled safely
enough to be used as pre-silicon/pre-board engineering evidence.
"""

import argparse
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
DEFAULT_OUTPUT = HERE / "results" / "engineering_quality_gate.json"
DEFAULT_MARKDOWN = HERE / "results" / "ENGINEERING_QUALITY_GATE_CN.md"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def add(checks, name, status, evidence, action):
    checks.append({
        "name": name,
        "status": status,
        "evidence": evidence,
        "required_action": action,
    })


def close(a, b, tol=1e-6):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def replay_summary(policy):
    return load_json(HERE / "results" / "structural_replay" / policy /
                     "simgrid_replay_summary.json")


def run_gate():
    checks = []
    evidence_path = HERE / "results" / "structural_evidence.json"
    plan_path = (HERE.parent / "16_hardware_calibration" / "results" /
                 "rf880_fp16_union_measurement_plan" / "measurement_plan_summary.json")
    transfer_plan_path = plan_path.parent / "transfer_measurements.csv"
    structural = load_json(evidence_path)
    plan = load_json(plan_path)
    comparison = structural["full_dag_comparison"]

    add(
        checks,
        "evidence_is_not_labeled_as_hardware_prediction",
        "pass" if not structural["valid_for_target_prediction"] else "block",
        f"structural evidence valid_for_target_prediction={structural['valid_for_target_prediction']}",
        "Keep sensitivity and target-measured results in separate tables.",
    )
    add(
        checks,
        "ffn_regions_are_dependency_certified",
        "pass" if structural["ffn_regions"] == 12 and not structural["rejected_ffn_regions"] else "block",
        f"found {structural['ffn_regions']} FFN regions; rejected={len(structural['rejected_ffn_regions'])}",
        "Fix DAG extraction before using FFN-level conclusions.",
    )
    reductions = [row["logical_byte_reduction_ratio"] for row in structural["boundary_by_ffn"]]
    add(
        checks,
        "ffn_fusion_has_structural_transfer_reason",
        "pass" if reductions and all(close(value, 0.8) for value in reductions) else "block",
        "all FFN regions reduce split FFN boundary bytes by 80%",
        "Recompute the derivation from actual edge sizes.",
    )
    gpu = comparison["gpu_only"]["makespan_us"]
    split = comparison["split_two_linears_fpga_gelu_gpu"]["makespan_us"]
    fused = comparison["declared_capability_auto"]["makespan_us"]
    no_full = comparison["no_full_ffn_kernel_auto"]["makespan_us"]
    add(
        checks,
        "split_linear_offload_is_rejected_by_cost_model",
        "pass" if split > gpu else "block",
        f"GPU-only {gpu / 1000:.3f} ms; split FFN linears {split / 1000:.3f} ms",
        "Do not offload singleton FFN Linears unless measured costs change.",
    )
    add(
        checks,
        "full_ffn_is_beneficial_only_under_declared_assumption",
        "pass" if fused < gpu and close(no_full, gpu) else "block",
        f"full FFN {fused / 1000:.3f} ms; no full-FFN kernel {no_full / 1000:.3f} ms",
        "Verify the FPGA full-FFN/GELU kernel before claiming speedup.",
    )

    expected = {
        "split_ffn_linears": (125, 96),
        "fixed_ffn": (101, 48),
        "linear_only_auto": (125, 0),
    }
    replay_ok = True
    replay_notes = []
    for policy, (task_count, transfer_count) in expected.items():
        summary = replay_summary(policy)
        policy_ok = (
            summary["task_count"] == task_count and
            summary["transfer_stage_count"] == transfer_count and
            summary["execution_mode"] == "simgrid_event_replay_sensitivity_parameters" and
            not summary["valid_for_target_prediction"]
        )
        replay_ok = replay_ok and policy_ok
        replay_notes.append(
            f"{policy}: tasks={summary['task_count']}, transfers={summary['transfer_stage_count']}"
        )
    add(
        checks,
        "simgrid_replay_matches_expected_task_and_transfer_counts",
        "pass" if replay_ok else "block",
        "; ".join(replay_notes),
        "Regenerate event replay before using timeline or latency numbers.",
    )

    add(
        checks,
        "measurement_plan_covers_fused_and_original_dags",
        "pass" if plan["tasks"] == 137 and plan["fpga_rows"] == 36 and plan["activation_bytes_per_element_planned"] == 2 else "block",
        f"tasks={plan['tasks']}, fpga_rows={plan['fpga_rows']}, activation_bytes={plan['activation_bytes_per_element_planned']}",
        "Regenerate the union measurement plan for the exact model precision.",
    )
    transfer_text = transfer_plan_path.read_text(encoding="utf-8")
    size_ok = "302592" in transfer_text and "1210368" in transfer_text
    add(
        checks,
        "measurement_plan_includes_key_ffn_payload_sizes",
        "pass" if size_ok else "block",
        "302592 B boundary activations and 1210368 B expanded FFN activations are planned",
        "Add missing FFN payload sizes to transfer calibration.",
    )
    add(
        checks,
        "heldout_end_to_end_validation_is_planned_but_not_completed",
        "warn" if plan["status"] == "waiting_for_target_hardware" else "pass",
        f"measurement plan status={plan['status']}; heldout rows={plan['heldout_end_to_end_template_rows']}",
        "Collect held-out board measurements before target-performance claims.",
    )

    blocked = [check for check in checks if check["status"] == "block"]
    warnings = [check for check in checks if check["status"] == "warn"]
    return {
        "status": "pass_with_hardware_warnings" if not blocked else "blocked",
        "software_ready_for_hardware_calibration": not blocked,
        "target_performance_claim_ready": False,
        "why_target_claim_not_ready": [
            "FPGA full-FFN/GELU kernel has not been measured on RF880.",
            "Jetson GPU-only baseline has not been measured under the final runtime and precision.",
            "PCIe/DMA path, fixed latency, bandwidth, and overlap have not been calibrated on the board.",
            "Held-out end-to-end hardware validation has not been collected.",
        ],
        "checks": checks,
        "summary": {
            "passes": sum(check["status"] == "pass" for check in checks),
            "warnings": len(warnings),
            "blocks": len(blocked),
        },
    }


def render_markdown(result):
    lines = [
        "# 工程质量门禁结果",
        "",
        f"- 总状态：`{result['status']}`",
        f"- 是否可以进入硬件标定阶段：`{result['software_ready_for_hardware_calibration']}`",
        f"- 是否可以声称真实硬件加速：`{result['target_performance_claim_ready']}`",
        "",
        "## 结论",
        "",
        "当前项目的软件建模、FFN 结构证据、SimGrid 事件回放和硬件测量计划已经可以作为上板前工程依据。"
        "但现阶段仍不能声称 RF880-AGX 板卡上已经获得端到端加速，因为关键算子、PCIe/DMA 路径和整模型结果还没有实测。",
        "",
        "## 检查项",
        "",
    ]
    for check in result["checks"]:
        lines.extend([
            f"- `{check['status']}` {check['name']}",
            f"  证据：{check['evidence']}",
            f"  后续动作：{check['required_action']}",
        ])
    lines.extend([
        "",
        "## 不能提前下结论的原因",
        "",
    ])
    for item in result["why_target_claim_not_ready"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    result = run_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "software_ready_for_hardware_calibration": result["software_ready_for_hardware_calibration"],
        "target_performance_claim_ready": result["target_performance_claim_ready"],
        "summary": result["summary"],
        "output": str(args.output),
        "markdown": str(args.markdown),
    }, indent=2))


if __name__ == "__main__":
    main()
