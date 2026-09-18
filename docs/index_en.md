English | [中文](index.md)

## PSuM Documentation Index

PSuM documentation is organized into three layers by reading depth. For first-time users, we recommend running the examples first, then coming back to read the concepts and design documents.

### 1. Quick Start

- [Docker Installation](../docker_install/readme_en.md): The recommended way to set up your environment, suitable for personal computers, workstations, or other controlled environments.
- [Dependency Installation](dependency_en.md) and [Environment Configuration & Building](building_en.md): Read these if Docker is not applicable; the content focuses more on environment and toolchain details.
- [Examples Overview](examples_en.md): Introduces the interactive tour examples under `example/` and the benchmark cases under `application/benchmark/`.

### 2. Learning to Use PSuM

- [Introduction to PIC](pic_en.md): Basic background on the Particle-in-Cell method.
- [C++ in PSuM](cpp-in-psum_en.md): Supplementary material on C++ templates, lambdas, and SYCL usage.
- [Programming Guide](programming_en.md): A systematic overview of PSuM's core API, including tags, particle containers, fields, boundaries, solvers, and more.

### 3. Understanding the Background & Contributing

- [Framework Design](design_en.md): Design notes for contributors and advanced users.
- `docs/detailed_design/`: Detailed design documents for complex modules, such as solver backends.
- [Code Style](coding-style_en.md): Recommended reading before contributing to development.
