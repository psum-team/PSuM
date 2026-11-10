#!/usr/bin/env python3
"""Render the add_back trajectory frames into a comparison GIF.

Left to right: 1st/2nd/3rd order deposition of the same trajectory point,
shared color scale, white background.
"""
import glob
import io
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from PIL import Image

PANELS = [("o1", "1st order (linear)"),
          ("o2", "2nd order (quadratic)"),
          ("o3", "3rd order (cubic)")]


def load_frame(path):
    lines = open(path).read().splitlines()
    dims = None
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("zone"):
            dims = [int(d) for d in re.findall(r"[ijk]=(\d+)", line)]
            start = i + 1
            break
    vals = np.array([t for l in lines[start:] for t in l.split()], dtype=np.float64)
    nx = dims[1]  # zone prints i = ny, j = nx; F=point rows: outer x, inner y
    ny = dims[0]
    pts = vals[:nx * ny * 3].reshape(-1, 3)
    return pts[:, 2].reshape(nx, ny).T


def main():
    frames = {tag: sorted(glob.glob(f"traj_{tag}_f*.plt")) for tag, _ in PANELS}
    n = min(len(v) for v in frames.values())
    if n == 0:
        print("plot_traj_gif: no frames found")
        return
    grids = {tag: [load_frame(p) for p in frames[tag][:n]] for tag, _ in PANELS}
    vmax = max(g.max() for tag, _ in PANELS for g in grids[tag])

    images = []
    for i in range(n):
        # fresh figure per frame (never stack artists across frames)
        fig, axes = plt.subplots(1, len(PANELS), figsize=(10, 3.6), dpi=100,
                                 constrained_layout=True)
        for ax, (tag, title) in zip(axes, PANELS):
            im = ax.imshow(grids[tag][i], origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
                           cmap="magma", norm=PowerNorm(0.45, vmin=0.0, vmax=vmax),
                           interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im, ax=list(axes), shrink=0.85, label="deposited weight")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight")
        plt.close(fig)
        images.append(Image.open(buf).convert("RGB"))

    out = "traj_compare.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=80, loop=0)
    print(f"plot_traj_gif: {out} ({n} frames, {len(PANELS)} panels, vmax={vmax:.3g})")


if __name__ == "__main__":
    main()
