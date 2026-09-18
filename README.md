language: English | [中文](README_cn.md)

# PSuM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PSuM is a C++/SYCL framework designed for the PIC (Particle-in-Cell) method. It aims to express particle, field, boundary, and collision logic in a single codebase while adapting to different hardware such as CPUs and GPUs.

At present, PSuM is a framework library delivered in source form: functionality is provided as source code and compiled into your application together with this repository's `include/` and `src/` — PSuM is not distributed as a prebuilt binary or an installable C++ package.

Official website: <https://psum.space>


## Quick Start

If you have control over your environment (e.g., a personal computer or workstation), prefer Docker:

- [Docker Installation](docker_install/readme_en.md)

If Docker is not suitable, follow the manual setup guide:

- [Dependency Installation](docs/dependency_en.md)
- [Environment Setup and Building](docs/building_en.md)

Once configured, run the test script to verify your environment:

```bash
./test/runCheck.sh
```

By default this is a smoke run: every target is compiled and started, and any test still running after the (short) default timeout is reported as a timeout rather than a failure. For a strict full run, use:

```bash
TIMEOUT=20 RUNCHECK_STRICT=1 bash test/runCheck.sh
```

In CI, long-running targets listed in `test/rcheck_skip_targets.list` are skipped, and the remaining targets are expected to finish within the time limit.

## First Example

The [Examples Overview](docs/examples_en.md) introduces the built-in examples.
The Tour series focuses on "runnable, visible, understandable" rather than complete physics models.
The Benchmark series is better suited for developers familiar with the PIC method.

## Goals

PIC (Particle-in-Cell) is naturally parallel at the particle level and well suited to GPUs — a single high-end GPU can approach the throughput of around a hundred CPU cores on some PIC workloads. But writing PIC directly for GPUs brings real engineering cost: learning SYCL/CUDA, harder debugging, memory layouts that work on CPU but not on GPU, and high migration cost across hardware vendors. PSuM aims to preserve the performance potential of GPUs while reducing this engineering complexity. It is built on SYCL, so the same C++ code can target CPUs, GPUs, and other platforms.

- **Shield hardware differences** — the goal is not merely to "make PIC run on GPUs," but to let PIC program expression stay as independent of specific hardware as possible. Users describe particle, field, boundary, and collision logic in one codebase; the framework shields part of the underlying hardware differences.
- **Reduce the complexity of high-performance implementations** — performance bottlenecks in PIC are not always in particle pushing or interpolation; they can also arise in Poisson equations, implicit Maxwell equations, or other linear system solves. PSuM separates the physical model from specific linear algebra implementations through pluggable solver backends, so users can choose appropriate backends based on hardware and problem scale. The framework also integrates high-performance intersection detection, collision pair generation, and similar algorithms.
- **Expose complexity in layers** — PSuM does not assume every user needs to understand the framework internals. Documentation and interfaces are roughly divided into three layers: the **application layer** runs examples or benchmark cases directly to get verifiable results; the **algorithm layer** composes particle containers, fields, boundaries, solvers, and collision modules to write custom cases; the **core layer** replaces or extends internal mechanisms such as particle boundary handling, collision models, and solver backends.

## Documentation Map

- To get started quickly: read the [Examples Overview](docs/examples_en.md) and follow the Tour series to modify and run code step by step.
- To write your own program: read the [Programming Guide](docs/programming_en.md) to learn about tags, particle containers, fields, solvers, and boundary systems.
- To understand PIC background: read the [PIC Introduction](docs/pic_en.md).
- To contribute to the framework: read the [Framework Design](docs/design_en.md) and design documents under `docs/detailed_design/`.
- The helper script `repo_heatmap.py` renders a code-distribution heatmap of `src/` (dependencies: pandas, plotly, matplotlib; PNG export additionally needs Kaleido).
For a complete navigation, see the [PSuM Documentation Index](docs/index_en.md).

## Contributing

PSuM is developed primarily by the core team. The best way to contribute is by
opening Issues — bug reports, feature suggestions, and design discussions are
all welcome. We are not yet set up to accept unsolicited pull requests; please
see the [Contributing Guide](CONTRIBUTING.md) for details.

## Citing PSuM

If you use PSuM in your research, please cite it via the metadata in
[CITATION.cff](CITATION.cff).

## License

PSuM is released under the [MIT License](LICENSE). Third-party components
bundled in `third_party/` retain their own licenses — see
[Third-Party Notices](THIRD_PARTY_NOTICES.md) for details.
