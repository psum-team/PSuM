#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
import numpy as np

EPS0 = 8.8541878128e-12
ELEMENTARY_CHARGE = 1.602176634e-19
DERIVED_VARIABLES = {"debye_length", "lambda_D", "lambda_debye"}
NODE_VARS = {"x", "y"}


def read_gamma(input_dir: Path) -> float:
    for path in (input_dir.parent / "config.json", input_dir / "config.json"):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as fp:
            gamma = float(json.load(fp).get("gamma", 1.0))
        return gamma if gamma > 0.0 else 1.0
    return 1.0


def parse_variables(line: str):
    if not line.lower().startswith("variables="):
        raise ValueError("data_out file does not start with a variables= header")
    names = line.split("=", 1)[1].strip()
    return [name.strip().strip('"') for name in names.split(",")]


def parse_zone(line: str):
    match_i = re.search(r"\bi\s*=\s*(\d+)", line, flags=re.IGNORECASE)
    match_j = re.search(r"\bj\s*=\s*(\d+)", line, flags=re.IGNORECASE)
    if not match_i or not match_j:
        raise ValueError(f"could not parse zone dimensions from: {line.strip()}")
    return int(match_i.group(1)), int(match_j.group(1))


def parse_cell_centered_vars(zone_line: str):
    ranges = set()
    for match in re.finditer(r"[\[(](\d+)(?:-(\d+))?[\])]=CELLCENTERED", zone_line, re.IGNORECASE):
        lo = int(match.group(1))
        hi = int(match.group(2)) if match.group(2) else lo
        for v in range(lo, hi + 1):
            ranges.add(v - 1)
    return ranges


def iter_from_name(path: Path):
    match = re.search(r"data_out_(\d+)\.plt$", path.name)
    return int(match.group(1)) if match else -1


def calculate_debye_length(data, variables, gamma=1.0):
    required = ("ne", "Te_parallel", "Te_perp")
    missing = [name for name in required if name not in variables]
    if missing:
        raise ValueError(f"debye_length needs variables: {', '.join(required)}; missing: {', '.join(missing)}")

    ne = data[:, variables.index("ne")]
    te_parallel = data[:, variables.index("Te_parallel")]
    te_perp = data[:, variables.index("Te_perp")]
    te = (te_parallel + 2.0 * te_perp) / 3.0

    debye = np.full_like(ne, np.nan, dtype=float)
    valid = (ne > 0.0) & (te > 0.0) & np.isfinite(ne) & np.isfinite(te)
    epsilon_eff = gamma * gamma * EPS0
    debye[valid] = np.sqrt(epsilon_eff * te[valid] / (ELEMENTARY_CHARGE * ne[valid]))
    return debye


def read_data_out_fields(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        variables = parse_variables(fp.readline().strip())
        zone_line = fp.readline().strip()
        while zone_line == "":
            zone_line = fp.readline().strip()

        ni, nj = parse_zone(zone_line)
        node_size = ni * nj
        cell_size = (ni - 1) * (nj - 1)
        cell_centered = parse_cell_centered_vars(zone_line)
        raw = fp.read()

    tokens = np.fromiter((float(t) for t in raw.split()), dtype=np.float64, count=len(raw.split()))
    is_block = "f=block" in zone_line.replace(" ", "").lower()

    if is_block:
        expected = sum(cell_size if vi in cell_centered else node_size for vi in range(len(variables)))
        if tokens.size != expected:
            raise ValueError(f"{path} has {tokens.size} values, expected {expected}")

        offset = 0
        blocks = {}
        for vi, name in enumerate(variables):
            size = cell_size if vi in cell_centered else node_size
            values = tokens[offset : offset + size]
            offset += size
            if vi in cell_centered:
                blocks[name] = values.reshape(nj - 1, ni - 1).T
            else:
                blocks[name] = values.reshape(nj, ni).T
    else:
        data = tokens.reshape(node_size, len(variables))
        blocks = {name: data[:, vi].reshape(nj, ni).T for vi, name in enumerate(variables)}

    x_nodes = blocks["x"][0, :]
    y_nodes = blocks["y"][:, 0]
    x = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    y = 0.5 * (y_nodes[:-1] + y_nodes[1:])

    fields = {}
    for name, values in blocks.items():
        if name in NODE_VARS:
            continue
        if values.shape == (ni - 1, nj - 1):
            fields[name] = values
        else:
            fields[name] = 0.25 * (values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:] + values[1:, 1:])
    return x, y, fields, variables


def read_frame(path: Path, var_name: str, gamma=1.0):
    x, y, fields, variables = read_data_out_fields(path)
    is_derived = var_name in DERIVED_VARIABLES
    if not is_derived and var_name not in fields:
        available = ", ".join(fields)
        derived = ", ".join(sorted(DERIVED_VARIABLES))
        raise ValueError(f"variable '{var_name}' not found. Available variables: {available}. Derived: {derived}")

    if is_derived:
        data = np.column_stack([fields[name].ravel() for name in variables if name not in NODE_VARS])
        field_names = [name for name in variables if name not in NODE_VARS]
        z = calculate_debye_length(data, field_names, gamma).reshape(fields["ne"].shape)
    else:
        z = fields[var_name]
    return x, y, z


def load_frames(input_dir: Path, var_name: str, gamma=1.0):
    files = sorted(input_dir.glob("data_out_*.plt"), key=iter_from_name)
    if not files:
        raise FileNotFoundError(f"no data_out_*.plt files found in {input_dir}")

    frames = []
    xs = ys = None
    for file in files:
        x, y, z = read_frame(file, var_name, gamma)
        if xs is None:
            xs, ys = x, y
        frames.append((iter_from_name(file), z))
    return xs, ys, frames


def finite_minmax(frames, percentile=None, positive_only=False):
    chunks = []
    for _, z in frames:
        values = z[np.isfinite(z)]
        if positive_only:
            values = values[values > 0.0]
        if values.size:
            chunks.append(values.ravel())
    if not chunks:
        return (1.0e-30, 1.0) if positive_only else (0.0, 1.0)
    values = np.concatenate(chunks)
    if values.size == 0:
        return 0.0, 1.0
    if percentile is None:
        return float(values.min()), float(values.max())
    lo = (100.0 - percentile) * 0.5
    hi = 100.0 - lo
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def make_discrete_boundaries(vmin, vmax, levels, scale):
    if levels < 2:
        raise ValueError("--levels must be at least 2")
    if scale == "log":
        return np.geomspace(vmin, vmax, levels + 1)
    return np.linspace(vmin, vmax, levels + 1)


def main():
    parser = argparse.ArgumentParser(description="Animate a variable from magnet-nozzle-2d data_out_*.plt files.")
    parser.add_argument(
        "variable",
        help="Variable name: phi, Bx, Br, ni, ne, vix, viy, viz, vex, vey, vez, "
        "Ti_parallel, Ti_perp, Te_parallel, Te_perp, debye_length "
        "(gamma-effective if config gamma is present)",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="case_test/output",
        help="Default: case_test/output",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Default: <input-dir>/<variable>.gif",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second. Default: 2")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap. Default: viridis")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum")
    parser.add_argument(
        "--levels",
        type=int,
        default=12,
        help="Number of discrete color levels for debye_length. Default: 12",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Use a continuous color scale instead of the default discrete debye_length scale.",
    )
    parser.add_argument(
        "--aspect",
        choices=("equal", "auto"),
        default="equal",
        help="Axis aspect ratio. Default: equal, so x and r use the same physical scale.",
    )
    parser.add_argument(
        "--scale",
        choices=("auto", "linear", "log"),
        default="auto",
        help="Color scale. Default: auto, which uses log for ni/ne and linear otherwise.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Auto color scale percentile, use 100 for full range. Default: 99",
    )
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI. Default: 140")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output) if args.output else input_dir / f"{args.variable}.gif"

    gamma = read_gamma(input_dir)
    x, y, frames = load_frames(input_dir, args.variable, gamma)
    scale = args.scale
    if scale == "auto":
        scale = "log" if args.variable in {"ni", "ne"} else "linear"
    use_discrete = args.variable in DERIVED_VARIABLES and not args.continuous

    percentile = None if args.percentile >= 100 else args.percentile
    auto_vmin, auto_vmax = finite_minmax(frames, percentile=percentile, positive_only=(scale == "log"))
    vmin = args.vmin if args.vmin is not None else auto_vmin
    vmax = args.vmax if args.vmax is not None else auto_vmax
    cmap = plt.get_cmap(args.cmap, args.levels if use_discrete else None)
    boundaries = None
    if scale == "log":
        vmin = max(vmin, 1.0e-300)
        if vmax <= vmin:
            vmax = vmin * 10.0
        frames = [(iter_num, np.ma.masked_less_equal(z, 0.0)) for iter_num, z in frames]
        if use_discrete:
            boundaries = make_discrete_boundaries(vmin, vmax, args.levels, scale)
            norm = BoundaryNorm(boundaries, cmap.N)
        else:
            norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        if use_discrete:
            if vmax <= vmin:
                vmax = vmin + 1.0
            boundaries = make_discrete_boundaries(vmin, vmax, args.levels, scale)
            norm = BoundaryNorm(boundaries, cmap.N)
        else:
            norm = None

    fig, ax = plt.subplots(figsize=(10, 4.8))
    mesh = ax.pcolormesh(
        x,
        y,
        frames[0][1],
        shading="auto",
        cmap=cmap,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        norm=norm,
    )
    cbar_kwargs = {}
    if boundaries is not None:
        cbar_kwargs["boundaries"] = boundaries
        cbar_kwargs["ticks"] = boundaries
        cbar_kwargs["spacing"] = "uniform"
    cbar = fig.colorbar(mesh, ax=ax, **cbar_kwargs)
    display_variable = args.variable
    if args.variable in DERIVED_VARIABLES:
        display_variable = f"{args.variable} (gamma-effective, gamma={gamma:g})"
    cbar.set_label(display_variable)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("r (m)")
    title = ax.set_title(f"Magnet nozzle 2D - {display_variable}, iter={frames[0][0]}")
    ax.set_aspect(args.aspect, adjustable="box")

    def update(frame):
        iter_num, z = frame
        mesh.set_array(z.ravel())
        title.set_text(f"Magnet nozzle 2D - {display_variable}, iter={iter_num}")
        return mesh, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000.0 / args.fps,
        blit=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".mp4":
        writer = animation.FFMpegWriter(fps=args.fps)
    else:
        writer = animation.PillowWriter(fps=args.fps)
    anim.save(output, writer=writer, dpi=args.dpi)
    plt.close(fig)
    discrete_msg = f", levels={args.levels}" if use_discrete else ""
    print(f"Wrote {output} from {len(frames)} frames, scale={scale}{discrete_msg}, vmin={vmin:g}, vmax={vmax:g}")


if __name__ == "__main__":
    main()
