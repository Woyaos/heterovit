import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATH = ROOT / "simgrid_baseline" / "12_full_dag_baselines" / "run_policy.py"
SPEC = importlib.util.spec_from_file_location("hetero_runtime", RUNTIME_PATH)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def main():
    profile = {
        "profile_kind": "target_measurement",
        "status": "complete",
        "valid_for_target_prediction": True,
        "activation": {"precision": "TEST", "bytes_per_element": 4},
        "task_costs_us": {"linear": {"GPU": 10.0, "FPGA": 4.0}},
        "communication_path": {
            "kind": "shared_mapped_one_copy",
            "host_fpga_pcie": {"fixed_latency_us": 2.0, "effective_bandwidth_GBps": 1.0},
        },
        "activation_compression": {
            "ratio": 1.0,
            "encode_fixed_us": 0.0,
            "decode_fixed_us": 0.0,
            "encode_bandwidth_GBps": 0.0,
            "decode_bandwidth_GBps": 0.0,
        },
        "pipeline": {"fpga_input_buffers": 2},
    }
    task = {
        "id": "linear",
        "task_type": "linear_projection",
        "input_shape_batch1": [1, 1, 4],
        "output_shape_batch1": [1, 1, 4],
    }
    model = runtime.SensitivityCostModel(profile)
    assert model.latency_us(task, "GPU") == 10.0
    assert model.latency_us(task, "FPGA") == 4.0
    assert model.stage_latency_us("host_fpga_pcie", 1000) == 3.0
    print("measured-cost profile path: PASS")


if __name__ == "__main__":
    main()
