import csv
import json
import math
import sys
from pathlib import Path


NUMERIC_COST_FIELDS = ["latency_mean_us", "latency_p50_us", "latency_p95_us", "samples"]
TEXT_COST_FIELDS = ["precision", "source_kind", "source_ref", "hardware"]
ALLOWED_SOURCE_KINDS = {"measured", "hls_report", "published", "sensitivity"}

PLATFORM_REQUIRED_PATHS = [
    "devices.GPU.model",
    "devices.GPU.power_mode",
    "devices.GPU.software_stack",
    "devices.FPGA.model",
    "devices.FPGA.bitstream_or_design",
    "devices.FPGA.clock_mhz",
    "communication.path",
    "communication.GPU_to_FPGA.fixed_latency_us",
    "communication.GPU_to_FPGA.effective_bandwidth_GBps",
    "communication.GPU_to_FPGA.samples",
    "communication.GPU_to_FPGA.source_kind",
    "communication.GPU_to_FPGA.source_ref",
    "communication.FPGA_to_GPU.fixed_latency_us",
    "communication.FPGA_to_GPU.effective_bandwidth_GBps",
    "communication.FPGA_to_GPU.samples",
    "communication.FPGA_to_GPU.source_kind",
    "communication.FPGA_to_GPU.source_ref",
    "communication.supports_bidirectional_overlap",
    "communication.supports_compute_transfer_overlap",
]


def nested_value(data, dotted_path):
    current = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def positive_number(value):
    try:
        number = float(value)
        return math.isfinite(number) and number > 0
    except (TypeError, ValueError):
        return False


def validate_cost_rows(costs_path):
    rows = list(csv.DictReader(costs_path.open(encoding="utf-8")))
    invalid_rows = []
    source_counts = {}
    for row in rows:
        missing = [field for field in TEXT_COST_FIELDS if not row.get(field)]
        missing += [field for field in NUMERIC_COST_FIELDS if not positive_number(row.get(field))]
        source_kind = row.get("source_kind")
        if source_kind and source_kind not in ALLOWED_SOURCE_KINDS:
            missing.append("source_kind_invalid")
        if missing:
            invalid_rows.append({"task_id": row.get("task_id"), "device": row.get("device"), "fields": missing})
        if source_kind:
            source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
    return rows, invalid_rows, source_counts


def main():
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            f"Usage: python {sys.argv[0]} <task_costs.csv> <platform_profile.json> [--strict]"
        )

    costs_path = Path(sys.argv[1]).resolve()
    platform_path = Path(sys.argv[2]).resolve()
    strict = len(sys.argv) == 4 and sys.argv[3] == "--strict"

    rows, invalid_rows, source_counts = validate_cost_rows(costs_path)
    platform = json.loads(platform_path.read_text(encoding="utf-8"))
    missing_platform_fields = [
        path for path in PLATFORM_REQUIRED_PATHS if nested_value(platform, path) is None
    ]
    valid = not invalid_rows and not missing_platform_fields
    report = {
        "valid_for_simulation": valid,
        "cost_rows": len(rows),
        "complete_cost_rows": len(rows) - len(invalid_rows),
        "incomplete_cost_rows": len(invalid_rows),
        "source_counts": source_counts,
        "missing_platform_fields": missing_platform_fields,
        "first_five_incomplete_rows": invalid_rows[:5],
    }
    print(json.dumps(report, indent=2))
    if strict and not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
