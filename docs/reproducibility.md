# Reproducibility and evidence policy

## Environment levels

The repository supports progressively stronger environments:

1. **Standard-library validation:** partition, evidence, calibration, streaming,
   and GPU-baseline unit tests.
2. **Analytical experiment:** deterministic cost-model runs without SimGrid.
3. **SimGrid replay:** Linux/WSL with compatible Python SimGrid bindings; one
   engine per process.
4. **IRIS smoke test:** a CPU/OpenMP IRIS build validates policy registration,
   metadata, and dependencies.
5. **Target evaluation:** Jetson AGX + RF880, real kernels, complete measured
   profile, and held-out end-to-end runs.

## Core validation

Run from the repository root:

```bash
python -m unittest discover -s simgrid_baseline/18_auto_block_partition -p "test*.py" -v
python -m unittest discover -s simgrid_baseline/19_single_inference_tile_streaming -p "test*.py" -v
python -m unittest discover -s simgrid_baseline/20_strong_gpu_baseline -p "test*.py" -v
python -m unittest discover -s simgrid_baseline/16_hardware_calibration -p "test*.py" -v
```

The GitHub workflow runs the same 33 tests on Python 3.10–3.12.

## Recreate the partition artifacts

```bash
python simgrid_baseline/18_auto_block_partition/partition.py
python simgrid_baseline/18_auto_block_partition/sweep.py
python simgrid_baseline/18_auto_block_partition/structural_evidence.py
```

For SimGrid replay, use the matrix scripts in separate processes as described
in that experiment's README:

```bash
python simgrid_baseline/18_auto_block_partition/run_replay_matrix.py
python simgrid_baseline/18_auto_block_partition/run_online_matrix.py
```

## Hardware calibration

Follow `simgrid_baseline/16_hardware_calibration/README.md`. Keep raw local
measurements in its ignored `measurements/local/` directory until they have
been reviewed for machine identifiers and converted into a curated dataset.
The validator must build a complete profile without fallback assumptions.

Before making a target-performance claim, record:

- commit hash, model hash, precision, batch size, and software versions;
- GPU and FPGA clock/power modes and thermal state;
- warm-up policy, sample count, median and p95;
- directional transfer samples for multiple payload sizes;
- prediction error on end-to-end samples not used for calibration;
- a strong TensorRT GPU-only baseline under matching conditions.

## Interpreting checked-in results

Machine-readable metadata is authoritative. If prose and metadata disagree,
prefer the metadata and open an issue. In particular, `sensitivity`,
`analytical_fallback_no_simgrid`, and `valid_for_target_prediction=false` forbid
claims of measured target speedup.

Generated detailed timelines are retained only when an evidence test consumes
them or when they are a small, curated example. New experiments should commit
configuration, summaries, and validation logic rather than large result trees.
