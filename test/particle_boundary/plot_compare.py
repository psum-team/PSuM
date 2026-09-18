#!/usr/bin/env python3
import numpy as np
from PIL import Image
import sys
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_plt(file):
    with open(file) as f:
        lines = f.readlines()
    height, width = 0, 0
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('zone'):
            dims = re.findall(r'[ij]=(\d+)', line)
            height, width = int(dims[0]), int(dims[1])
            data_start = i + 1
            break
    values = np.fromstring(''.join(lines[data_start:]), sep='\t').reshape(-1, 5)
    r = values[:, 2].reshape(width, height).T
    g = values[:, 3].reshape(width, height).T
    b = values[:, 4].reshape(width, height).T
    return r, g, b, width, height

def plt_to_rgb(file):
    r, g, b, w, h = load_plt(file)
    gamma = 0.45
    rgb = np.stack([
        np.power(np.clip(r, 0, 1), gamma),
        np.power(np.clip(g, 0, 1), gamma),
        np.power(np.clip(b, 0, 1), gamma)
    ], axis=-1)
    return np.flipud(rgb)

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 plot_compare.py <baseline.plt> <udf.plt>")
        sys.exit(1)

    print("Loading images...")
    img0 = plt_to_rgb(sys.argv[1])
    img1 = plt_to_rgb(sys.argv[2])

    print("Computing metrics...")
    diff = np.abs(img0 - img1)
    diff_rgb = np.mean(diff, axis=2)
    mse = np.mean(diff ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    fig = plt.figure(figsize=(16, 5))

    gs = fig.add_gridspec(1, 3, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img0)
    ax0.set_title('Baseline', fontsize=14, fontweight='bold')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img1)
    ax1.set_title('UDF', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2])
    scale_factor = 10.0
    diff_display = np.clip(diff * scale_factor, 0, 1)
    im_diff = ax2.imshow(diff_display)
    ax2.set_title(f'Difference (x{int(scale_factor)})', fontsize=14, fontweight='bold')
    ax2.axis('off')

    plt.suptitle('UDF Ray Tracing Comparison', fontsize=16, fontweight='bold', y=0.95)

    fig.text(0.5, 0.05,
             f'$\\text{{MSE}} = {mse:.2e}$    $\\text{{PSNR}} = {psnr:.2f}\\ \\text{{dB}}$    $\\max|\\Delta| = {max_diff:.4f}$',
             ha='center', fontsize=14)

    output = "ray_comparison.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved to {output}")

    print(f"\nMSE: {mse:.6e}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Max diff: {max_diff:.4f}")
    print(f"Mean diff: {mean_diff:.4f}")

if __name__ == '__main__':
    main()
