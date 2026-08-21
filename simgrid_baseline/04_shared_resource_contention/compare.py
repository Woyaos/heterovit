import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULT_PATTERN = re.compile(r"RESULT requests=(\d+) makespan_ms=([0-9.]+)")


def run_case(request_count):
    completed = subprocess.run(
        [
            sys.executable,
            str(HERE / "contention_sim.py"),
            str(request_count),
            str(HERE / "platform.xml"),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    match = RESULT_PATTERN.search(completed.stdout)
    if not match:
        raise RuntimeError(f"No result found in output:\n{completed.stdout}")
    return float(match.group(2))


def main():
    one_request_ms = run_case(1)
    two_requests_ms = run_case(2)
    sequential_reference_ms = 2 * one_request_ms

    print("Shared GPU + FPGA + 4 GB/s PCIe")
    print(f"one request makespan       : {one_request_ms:.3f} ms")
    print(f"two concurrent makespan    : {two_requests_ms:.3f} ms")
    print(f"two sequential reference   : {sequential_reference_ms:.3f} ms")
    print(f"concurrent slowdown vs one : {two_requests_ms / one_request_ms:.3f}x")
    print(f"throughput, one request    : {1000 / one_request_ms:.3f} requests/s")
    print(f"throughput, two concurrent : {2000 / two_requests_ms:.3f} requests/s")


if __name__ == "__main__":
    main()
