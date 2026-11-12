#!/usr/bin/env python3
"""Render the add_back trajectory frames into a comparison GIF.

Left to right: 1st/2nd/3rd/4th order deposition of the same trajectory point,
shared color scale, white background. Also runs quantitative checks
(frame count per order, partition of unity per frame); exits non-zero on failure.
"""
import glob
import io
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from PIL import Image

PANELS = [("o1", "1st order (linear)"),
          ("o2", "2nd order (quadratic)"),
          ("o3", "3rd order (cubic)"),
          ("o4", "4th order (quartic)")]


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


def load_frame_xyz(path):
    """Same as load_frame but also returns the x and y node coordinates."""
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
    x = pts[:, 0].reshape(nx, ny)[:, 0]
    y = pts[:, 1].reshape(nx, ny)[0, :]
    w = pts[:, 2].reshape(nx, ny)
    return x, y, w


SUM_TOL = 5e-4   # plot files print ~6 significant digits
EXPECT_FRAMES = 65
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main():
    frames = {tag: sorted(glob.glob(f"traj_{tag}_f*.plt")) for tag, _ in PANELS}
    n = min(len(v) for v in frames.values())
    if n == 0:
        print("plot_traj_gif: no frames found")
        sys.exit(1)
    grids = {tag: [load_frame(p) for p in frames[tag][:n]] for tag, _ in PANELS}

    # quantitative checks (CI-valid): frame count per order and per-frame sums
    for tag, _ in PANELS:
        check(len(frames[tag]) == EXPECT_FRAMES,
              f"traj {tag}: {len(frames[tag])} frames (expect {EXPECT_FRAMES})")
        sums = [g.sum() for g in grids[tag]]
        worst = max(abs(np.array(sums) - 1.0))
        check(worst < SUM_TOL, f"traj {tag}: all frame sums == 1 (worst {worst:.2e})")

    # support nesting: same frame, the (k+1)-order support must contain the
    # k-order support ("one ring larger"); node count capped by (order+1)^2
    W = 1e-6   # below print precision: treat as zero
    tags = [tag for tag, _ in PANELS]
    bad_nest = bad_cap = 0
    for i in range(n):
        masks = [grids[t][i] > W for t in tags]
        for lo, hi in zip(masks, masks[1:]):
            if (lo & ~hi).any():
                bad_nest += 1
        for (t, _), m in zip(PANELS, masks):
            if m.sum() > (int(t[1]) + 1) ** 2:
                bad_cap += 1
    check(bad_nest == 0, f"support nesting o1<=o2<=o3<=o4 every frame ({bad_nest} violations)")
    check(bad_cap == 0, f"support size <= (order+1)^2 every frame ({bad_cap} violations)")

    # centroid back-inference: the deposit's first moment must agree across
    # orders (each order's centroid reconstructs the same particle position)
    CENT_TOL = 1e-4
    worst_c = 0.0
    for i in range(n):
        cents = []
        for t in tags:
            x, y, w = load_frame_xyz(frames[t][i])
            ww = w.sum()
            cents.append((float((w * x[:, None]).sum() / ww), float((w * y[None, :]).sum() / ww)))
        for a, b in zip(cents, cents[1:]):
            worst_c = max(worst_c, abs(a[0] - b[0]), abs(a[1] - b[1]))
    check(worst_c < CENT_TOL, f"centroid agreement across orders (worst {worst_c:.2e})")

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
    print(f"plot_traj_gif: {'ALL PASS' if not failures else str(len(failures)) + ' FAILURES'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
