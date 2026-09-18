# collision-less-flow

A 2D collisionless magnetized-plasma flow in a magnetic nozzle with an
analytic coil field. Ions and electrons are loaded from a reservoir, accelerated
by the self-consistent field plus the external coil field, and the plume is
observed downstream. This example exercises the electrostatic solver stack
(finite-difference / sparse-LU backends), particle push, absorbing and
reflecting boundaries, and diagnostics output.

## Layout

- `main.cpp` — simulation entry point (reads `config.json`, builds and runs the case)
- `config.json` — case configuration (domain, plasma source, coil field, output cadence)
- `*.hpp` — case modules: species setup, field operations, moments/diagnostics, checkpoints
- `plot_record.py` — plot the monitor curves (`record.plt`)
- `animate_dataout.py <var>` — build a GIF animation of a field variable from `data_out_*.plt`
- `read_checkpoint.py` — inspect a `checkpoint*.bin` file

## Run

```bash
make main      # build (requires the PSuM build environment; see docs/building*.md)
make run       # run in ./case_test with the default config.json
```

Progress is written to `case_test/run.log`; field snapshots go to
`case_test/output/data_out_<step>.plt`, monitor samples to `case_test/output/record.plt`,
and checkpoints to `case_test/output/checkpoint_<step>.bin`.

## Visualize

```bash
cd case_test
python3 ../plot_record.py -i output -o monitor.png
python3 ../animate_dataout.py phi -i output -o phi.gif
```

`animate_dataout.py` supports the variables written in the frames, e.g.
`phi`, `Bx`, `Br`, `ni`, `ne`, velocity components (`vix`…`vez`), temperature
components (`Ti_parallel`, `Ti_perp`, `Te_parallel`, `Te_perp`) and
`debye_length`.

## Expected behaviour

Particle counts grow from the source and saturate as losses balance injection;
the electrostatic potential `phi` develops a steady nozzle profile. Runtime
scales with the step count in `config.json`; expect tens of minutes on a
workstation GPU with the shipped configuration.
