# Third-party notices

## IRIS

The `iris-main/` directory contains a vendored copy of the IRIS heterogeneous
runtime from <https://github.com/ORNL/iris>. IRIS is distributed under the
BSD 3-Clause License; its license text is retained at `iris-main/LICENSE`.
Project-specific integration code is concentrated in
`iris-main/apps/hetero_vit_runtime/`.

Additional license files shipped by IRIS and its headers remain in their
original locations. Those components are not relicensed by the root Apache-2.0
license.

## SimGrid

Some experiment paths can use the external SimGrid Python bindings. SimGrid is
not vendored by this repository. Users are responsible for installing a
compatible version and complying with its license.

## Models and hardware documentation

Model weights and vendor hardware manuals are not distributed by this project.
Derived DAG metadata and measurement templates must not be interpreted as a
grant of rights to any underlying model or documentation.
