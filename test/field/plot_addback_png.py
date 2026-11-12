#!/usr/bin/env python3
"""Render the higher-order add_back comparison plt files into summary PNGs."""
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    nx, ny = dims[1], dims[0]
    pts = vals[:nx * ny * 3].reshape(-1, 3)
    return pts[:, 2].reshape(nx, ny).T


SUM_TOL = 5e-4   # plot files print ~6 significant digits
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def render_group(prefix, stem_hi, stem_lo, ks, out):
    hi = load_frame(f"{prefix}_{stem_hi}.plt")
    lo = {k: load_frame(f"{prefix}_{stem_lo}_k{k}.plt") for k in ks}
    df = {k: load_frame(f"{prefix}_diff_k{k}.plt") for k in ks}

    # quantitative checks (CI-valid): partition of unity, non-negativity,
    # and strictly decreasing L2 of (hi - lo) as k grows
    check(abs(hi.sum() - 1.0) < SUM_TOL, f"{prefix}: hi deposit sum={hi.sum():.6f}")
    check(hi.min() >= 0.0, f"{prefix}: hi deposit non-negative (min={hi.min():.2e})")
    l2s = []
    for k in ks:
        check(abs(lo[k].sum() - 1.0) < SUM_TOL, f"{prefix}: k={k} lo deposit sum={lo[k].sum():.6f}")
        l2s.append(float(np.sqrt((df[k] ** 2).sum())))
    check(l2s[0] > l2s[1] > l2s[2],
          f"{prefix}: L2 diff strictly decreasing with k ({['%.2e' % v for v in l2s]})")

    vmax = max(hi.max(), max(v.max() for v in lo.values()))

    fig, axes = plt.subplots(2, len(ks) + 1, figsize=(3.8 * (len(ks) + 1), 7.2), dpi=120,
                             constrained_layout=True)
    im = axes[0, 0].imshow(hi, origin="lower", extent=(0, 1, 0, 1), cmap="magma",
                           vmin=0, vmax=vmax, interpolation="nearest")
    axes[0, 0].set_title(f"{stem_hi} order at p", fontsize=11)
    for col, k in enumerate(ks, start=1):
        axes[0, col].imshow(lo[k], origin="lower", extent=(0, 1, 0, 1), cmap="magma",
                            vmin=0, vmax=vmax, interpolation="nearest")
        axes[0, col].set_title(f"{stem_lo} order, k={k}", fontsize=11)
    axes[0, 0].set_ylabel("field")
    fig.colorbar(im, ax=list(axes[0]), shrink=0.85, label="deposited weight")

    axes[1, 0].set_visible(False)  # keep diffs aligned under their k columns
    for col, k in enumerate(ks, start=1):
        dm = np.abs(df[k]).max()  # each diff gets its own symmetric range
        imd = axes[1, col].imshow(df[k], origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm",
                                  vmin=-dm, vmax=dm, interpolation="nearest")
        axes[1, col].set_title(f"diff (k={k},\n±{dm:.2e})", fontsize=10)
        fig.colorbar(imd, ax=axes[1, col], shrink=0.75, label="hi - lo" if col == 1 else "")

    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"plot_addback_png: {out}")


if __name__ == "__main__":
    ks = [1, 8, 32]
    render_group("addback2", "2nd", "1st", ks, "addback2_summary.png")
    render_group("addback3", "3rd", "2nd", ks, "addback3_summary.png")
    render_group("addback4", "4th", "3rd", ks, "addback4_summary.png")
    print(f"plot_addback_png: {'ALL PASS' if not failures else str(len(failures)) + ' FAILURES'}")
    sys.exit(1 if failures else 0)
