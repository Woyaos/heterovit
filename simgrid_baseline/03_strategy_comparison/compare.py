import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULT_PATTERN = re.compile(r"RESULT strategy=(\S+) time_ms=([0-9.]+)")


def run_case(strategy, platform_name):
    command = [
        sys.executable,
        str(HERE / "strategy_sim.py"),
        strategy,
        str(HERE / platform_name),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    match = RESULT_PATTERN.search(completed.stdout)
    if not match:
        raise RuntimeError(f"No result found in output:\n{completed.stdout}")
    return float(match.group(2))


def main():
    cases = [
        ("slow-4GBps", "platform_slow_pcie.xml"),
        ("fast-16GBps", "platform_fast_pcie.xml"),
    ]

    print("PCIe scenario     GPU-only (ms)   Offload (ms)   Best")
    print("-" * 59)
    for label, platform in cases:
        gpu_time = run_case("gpu-only", platform)
        offload_time = run_case("offload", platform)
        best = "GPU-only" if gpu_time <= offload_time else "Offload"
        print(f"{label:<17} {gpu_time:>13.3f} {offload_time:>14.3f}   {best}")


if __name__ == "__main__":
    main()
