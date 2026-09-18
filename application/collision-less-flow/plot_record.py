#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INDEX_COLS = {"time", "i"}

PAIR_GROUPS = [
    ({"ele_num", "ion_num"}, "particle count"),
    ({"ion_input", "ele_input"}, "injected count"),
    ({"phi_min", "phi_max"}, "phi range (V)"),
]


def read_record(path: Path):
    if not path.exists():
        return [], {}
    columns = None
    all_data = {}
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            if line.startswith("variables"):
                columns = [name.strip().strip('"') for name in line.split("=", 1)[1].split(",")]
                continue
            parts = line.split()
            if columns is None:
                columns = [
                    "time",
                    "i",
                    "ele_num",
                    "ion_num",
                    "ion_input",
                    "ele_input",
                    "reflect_threshold",
                    "phi_min",
                    "phi_max",
                ]
            if len(parts) < len(columns):
                continue
            try:
                for idx, name in enumerate(columns):
                    all_data.setdefault(name, []).append(float(parts[idx]))
            except ValueError:
                continue
    if columns is None:
        return [], {}
    iters = all_data.pop("i", all_data.pop("time", []))
    data_cols = [c for c in columns if c not in INDEX_COLS and c in all_data]
    return iters, {c: all_data[c] for c in data_cols}


def build_groups(data_columns):
    groups = []
    used = set()
    for group_set, ylabel in PAIR_GROUPS:
        present = [c for c in data_columns if c in group_set]
        if present:
            groups.append((present, ylabel))
            used.update(present)
    for c in data_columns:
        if c not in used:
            groups.append(([c], c))
    return groups


def plot_record(record_path: Path, output_path: Path):
    iters, data = read_record(record_path)
    if not data:
        print(f"No usable rows in {record_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = build_groups(list(data.keys()))
    n_plots = len(groups)

    fig, axes = plt.subplots(n_plots, 1, figsize=(11, 3.2 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    fig.suptitle(f"Magnet nozzle 2D monitor: {record_path}", fontsize=12)

    for ax, (cols, ylabel) in zip(axes, groups):
        for col in cols:
            ax.plot(iters, data[col], label=col, linewidth=1.6)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if len(cols) > 1:
            ax.legend(loc="best")

    axes[-1].set_xlabel("iteration")

    n = len(iters)
    summary = f"{n} rows, iter {iters[0]:.0f} -> {iters[-1]:.0f}"
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(
        f"Wrote {output_path} with {n} rows "
        f"(iter {iters[0]:.0f} -> {iters[-1]:.0f})"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot magnet-nozzle-2d record.plt monitor curves.")
    parser.add_argument(
        "record",
        nargs="?",
        default=None,
        help="Path to record.plt. Defaults to <input-dir>/record.plt",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="case_test/output",
        help="Directory containing record.plt. Default: case_test/output",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Default: <record directory>/record_monitor.png",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Refresh interval in seconds. Omit or use 0 for one-shot plotting.",
    )
    args = parser.parse_args()

    record_path = Path(args.record) if args.record else Path(args.input_dir) / "record.plt"
    output_path = Path(args.output) if args.output else record_path.parent / "record_monitor.png"

    if args.watch <= 0:
        plot_record(record_path, output_path)
        return

    while True:
        plot_record(record_path, output_path)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
