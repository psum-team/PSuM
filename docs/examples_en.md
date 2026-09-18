English | [中文](examples.md)

## PSuM Examples

The example programs included with PSuM fall into two categories: the **tour series** and the **benchmark series**.

### Tour Series (Quick Start)

The tour series guided documentation is located in `docs/psum_tour/`, providing step-by-step instructions for writing tour examples from scratch.
Reference code for each stage is also provided in the corresponding subdirectory under `example/`, so you can compare with your own implementation.
These examples do not aim for physical rigor or completeness; instead, they emphasize **interactivity** and **visual verifiability** — you can tell at a glance from the visualization output whether the program behaves as expected.

When reading the tour documentation, it is recommended to create your own files and follow along with the instructions.
The files already present under `example/` (such as `stage_1.cpp`, `stage_2.cpp`) are "reference solutions" — you do not need to edit them; create your own file and compare against them.

- **Big Bang** (`example/big_bang/`, [details](psum_tour/big_bang_en.md)): A 2D electrostatic PIC introductory example.
A small cluster of same-sign charged particles is placed in a square domain. The self-consistent electric field causes the cluster to expand outward, and particles reaching the boundary are absorbed.
The documentation unfolds in 6 stages: from grid and field creation, particle initialization and charge deposition, to particle motion and boundary handling, Poisson solver integration, and finally energy conservation diagnostics.

- **Neon Light** (`example/neon_light/`, [details](psum_tour/neon_light_en.md)): A 2D Poisson complex boundary example.
Restores a `host_node_field2D` from a structured `.plt` bitmap, and uses `Dirichlet_func` and `interp` to set the pattern region as an internal time-varying Dirichlet boundary.
The documentation unfolds in 5 stages: from bitmap reading and Poisson solver setup, to internal boundary configuration, time-varying boundary values, and the final animation effect.

- **Telescope** (`example/telescope/`, [details](psum_tour/telescope_en.md)): A 2D ray tracing example.
Simulates a simplified Maksutov telescope: parallel light enters from the left, passes through the corrector via refraction, reflects off the primary mirror, then the secondary mirror, and converges onto the focal plane.
The documentation unfolds in 5 stages: from photon definition and basic motion loop, boundary system integration, to curved boundary surface reflection implementation, and finally adding refraction and the corrector, with animation output throughout.

- **Avalanche** (`example/avalanche/`, [details](psum_tour/avalanche_en.md)): A 2D MCC collision example.
Fast electrons are placed in a domain with a Gaussian gas cloud as the background. Electrons entering the cloud undergo elastic scattering and ionization collisions; secondary electrons continue to collide and ionize, forming an avalanche cascade.
The documentation unfolds in 5 stages: from particle initialization, fields and deposition, to MCC collision model integration, secondary particle production and the buffer mechanism.

> If you are interested in writing a new tour example: create the code directory under `example/` (with a makefile and per-stage sources), add matching bilingual guided documents under `docs/psum_tour/`, and index them in `docs/examples*.md`. New directories are expected to build cleanly and to keep the documentation links valid.

### Benchmark Series

Benchmark examples are located under `application/benchmark/`. They are the project's built-in benchmark cases used to exercise PSuM's functionality and performance, covering different solvers and physical scenarios.
Each case directory contains a brief README with its reference literature.

- **CCP** (`application/benchmark/CCP/`): Capacitively coupled plasma (1D), He discharge + MCC, RF bias 450 V / 13.56 MHz, with literature comparison.
- **DSMC** (`application/benchmark/DSMC/`): Rarefied gas DSMC example, N₂ supersonic jet impinging on a plate, VHS collision model + internal energy exchange.
- **EDI** (`application/benchmark/EDI/`): ExB drift discharge (2D), simulating electron/ion transport in a Hall thruster channel, with magnetic field profile and ionization source terms.
- **TSI** (`application/benchmark/TSI/`): Two-stream instability (1D), a classic electrostatic kinetic benchmark for verifying energy conservation and phase space evolution.
- **sheath** (`application/benchmark/sheath/`): Plasma sheath (1D), biased wall 12.5 V, with literature comparison.
- **particle-impulse-integration** (`application/benchmark/particle-impulse-integration/`): Particle impulse integration; a fast algorithm for free molecular flow steady state, compared against a high-accuracy reference solution.
