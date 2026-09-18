language: English | [中文](CONTRIBUTING_cn.md)

# Contributing to PSuM

PSuM is an active research project, currently developed primarily by the
Beihang JLPP team (Joint Laboratory of Plasma and Propulsion).

We look forward to feedback and discussion from the community.

The most helpful way to engage right now is by opening **Issues** — bug reports,
feature suggestions, and design discussions all help improve the project.

**We are not yet set up to accept unsolicited pull requests.** If you have a
specific change in mind, please start with an Issue so we can discuss whether
and how it fits.

This document explains how to set up a development environment and run the
tests, which is useful for anyone building on or studying PSuM — whether or not
you contribute back. It also documents the conventions the maintainers follow,
for reference.

PSuM is a C++20/SYCL framework for the [Particle-in-Cell (PIC)](docs/pic_en.md) method. It is
header-first, with two compiled components (field solver backends and the
`pypsum` Python bindings). For background, read the
[Programming Guide](docs/programming_en.md).

## Project layout

| Path | Purpose |
|------|---------|
| `src/<module>/` | Module implementations (headers). Each dir = one `psum::<module>` namespace. |
| `include/psum/<module>.hpp` | Public aggregation headers per module. |
| `test/<module>/makefile` | Per-module tests. |
| `application/` | Full physics applications (benchmark cases). |
| `example/` | Tutorial examples (Tour series). |
| `pypsum/` | Python interop / serialization module. |
| `third_party/` | Vendored: `json.hpp` (MIT). |

## Development environment setup

The build system is Make.
Dependencies: g++ (≥11), AdaptiveCpp (`acpp`), Eigen, optional CUDA/UMFPACK. Python components additionally require python3-venv (Debian/Ubuntu) or virtualenv — the build script creates a virtual environment automatically.
The repository provides a script such as `env_scan.sh` to help manage
environment variables.

```bash
bash env_scan.sh                 # auto-detect deps, generates env_load.sh
source env_load.sh               # sets PATH, CPATH, LIBRARY_PATH, LD_LIBRARY_PATH
bash build.sh                    # compiles field solver backends + pypsum, runs basic tests
```

- Override defaults in `config.mk.local` (gitignored), never edit `config.mk` directly.
- No CUDA? `echo "USE_CUDA := 0" > config.mk.local`
- `env_load.sh` is auto-generated and gitignored. It must be regenerated after
  dependency changes.

See [Dependency Installation](docs/dependency_en.md) and
[Environment Setup and Building](docs/building_en.md) for full details.

## Running tests

```bash
cd test && bash runCheck.sh                    # smoke test: compile+run all test/ targets
bash test/runCheck.sh ../application            # also works on application/
cd test/field_solver && make testpsolver2d      # single test target
```

- `runCheck.sh` env vars: `TIMEOUT` (default 1s), `JOBS` (default 8), `SKIP_DIRS`, `RUNCHECK_STRICT` (default 0; `1` = count timeouts as failures), `RUNCHECK_SKIP_TARGETS` (comma-separated `dir/target` list to skip, e.g. long-running tests), `RUNCHECK_SKIP_FILE` (path to a file listing skipped `dir/target` entries, one per line; `#` comments and blank lines are ignored; missing file is a hard error).
- Mass failures (>50%) almost always mean missing environment — run
  `source env_load.sh` first.
- Committing required data files (`.plt`, `.mas`, `.csv`, …): `.gitignore`
  ignores these formats globally. Place them under `reference_results/` or
  `necessary_input/` (both are exempted), or add an explicit `!` exception line
  to `.gitignore` for your path.

When writing new test targets under `test/<module>/`, note that `runCheck.sh`
parses `make -B -n` dry-run output, so:

1. The first recipe line must be the compiler command (no `@echo` before it).
2. The compile line must have an explicit `-o <output_file>`.
3. Target names `all`, `clean`, `test_cov`, `py_test` are filtered out — don't
   use them for real tests. Aggregate targets (e.g. `test_multigrid`) must be added
   to the same exclusion list, or runCheck will double-count their first dependency.
4. Don't use automatic variables (`$@`, `$<`) after `-o` — the dry-run won't
   expand them.

## Code style

PSuM follows a set of deliberate deviations from C++ defaults. Read
[`docs/coding-style_en.md`](docs/coding-style_en.md) in full before writing code.
Key points:

- **In most cases, snake_case**: files, classes, functions, namespaces.
- **Member variables and private methods**: trailing underscore (`data_`, `set_size_()`).
- **Enum values**: camelCase (`cellCentered`).
- **Template params conflicting with type aliases**: leading underscore (`_Dimension`, `_Scalar`).
- **Header guards**: `PSUM_<DIR>_<FILE>_HPP` (e.g. `PSUM_TAG_FOUNDATION_HPP`).
- **File→namespace mapping**: `src/<module>/<file>.hpp` → `namespace psum { namespace <module> { ... } }`.
- Never use `using namespace sycl;`.

## Reporting issues

The best way to contribute right now is by opening Issues:

- **Bug reports** — describe what you did, what you expected, and what happened
  instead. Include the command you ran, your environment (compiler, GPU, OS),
  and any error output. A minimal reproducer helps a lot.
- **Feature suggestions** — explain the use case and what you're trying to
  achieve, rather than just the solution you have in mind.
- **Design discussions** — for larger ideas, an Issue is a good place to talk
  through the approach before anything is built.

If you are thinking about a code change, please open an Issue to discuss it
first. This helps avoid duplicated effort and ensures the change fits the
project's direction.

## Conventions for changes

Please follow the conventions below when changing code. They also document how
the maintainers themselves work, for reference.

- **Write tests** for new functionality, placed under `test/<module>/` following
  the existing per-module makefile convention.
- **Run the test gate locally** before submitting:
  ```bash
  cd test && bash runCheck.sh
  ```
  The default invocation is the fast smoke tier (short timeouts, skipped long
  targets listed in `test/rcheck_skip_targets.list`). It must pass with zero
  failures. If your change touches long-running or heavy-load targets, also
  verify them locally with `TIMEOUT=20 RUNCHECK_STRICT=1 bash runCheck.sh`
  or by running the affected targets directly.
- **Keep changes focused.** One logical change per contribution makes review
  faster and history cleaner.
- **Match surrounding code.** Read a few neighboring files first and follow
  their naming, comment density, and idioms. Avoid reformatting unrelated code.

There is no Contributor License Agreement (CLA). Any contribution that is
accepted is licensed under the project's [MIT License](LICENSE).
