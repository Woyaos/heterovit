# Contributing

Contributions are welcome in the partitioner, cost model, replay validation,
calibration workflow, target kernels, and documentation.

## Before opening a pull request

1. Keep target measurements separate from sensitivity assumptions.
2. Add or update tests for scheduling, graph, profile, or evidence changes.
3. Run the four core validation commands from the root README.
4. Do not commit model weights, build products, raw hardware manuals, secrets,
   or large generated timeline trees.
5. State the model, precision, target, execution mode, and validity level for
   every new result.

## Change scope

Prefer small commits with one purpose. A change to an assumed parameter must
not be presented as an optimization result. Target-specific capabilities must
be represented explicitly and default to unverified until measurement or
implementation evidence is available.

## Bug reports

Include the command, Python version, operating system, commit hash, relevant
profile, and complete error output. For numerical disagreements, include the
expected evidence level and whether SimGrid or analytical fallback was used.
