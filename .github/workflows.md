# CI Workflows

> This table maps every workflow YAML in this directory to its trigger, scope,
> and gate semantics, so the whole pipeline can be seen at a glance. It is kept
> in sync with the workflow files.

| Workflow | File | Track | Trigger | Content | Depends on |
|---|---|---|---|---|---|
| ci | workflows/ci.yml | Fast functional gate | push(main) / PR | env discover → build.sh → runCheck (TIMEOUT=20, RUNCHECK_STRICT=1, RUNCHECK_SKIP_FILE=test/rcheck_skip_targets.list), plus a separate MPI multi-process job (2/3/4 ranks via test/mpi/run_mpi_tests.sh) | no_gpu container image |

The container image is built from `docker_install/dockerfile_no_gpu` and
published as `ghcr.io/psum-team/psum_env:no_gpu`.
