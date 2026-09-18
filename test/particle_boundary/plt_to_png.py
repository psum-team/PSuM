#!/usr/bin/env python3
"""Convert a PSuM Tecplot image dump (.plt) to PNG.
"""
import re
import sys
import numpy as np
from PIL import Image

GAMMA = 0.45


def parse_header(lines):
    variables = None
    dims = None
    cell_centered = False
    for i, line in enumerate(lines):
        if line.startswith('variables='):
            variables = line.split('=', 1)[1].strip().split(',')
        elif line.startswith('zone'):
            dims = [int(d) for d in re.findall(r'[ijk]=(\d+)', line)]
            cell_centered = 'CELLCENTERED' in line.upper()
            return variables, dims, cell_centered, i + 1
    return variables, dims, cell_centered, 0


def to_uint8(field):
    field = np.flipud(np.clip(field, 0.0, 1.0)) ** GAMMA * 255.0
    return field.astype(np.uint8)


def main():
    file = sys.argv[1] if len(sys.argv) > 1 else 'ray_tracing_img.plt'
    lines = open(file).read().splitlines()
    variables, dims, cell_centered, start = parse_header(lines)
    if not variables or not dims or len(dims) != 2:
        print(f"{file}: unsupported header (expected a 2D zone)")
        return 1

    vals = np.array([t for l in lines[start:] for t in l.split()], dtype=np.float64)
    node_count = dims[0] * dims[1]
    # zone prints i = cell_num[1]+1 then j = cell_num[0]+1
    cell0, cell1 = dims[1] - 1, dims[0] - 1
    cell_count = cell0 * cell1

    if not cell_centered:
        print(f"{file}: only F=block cell-centered zones are supported")
        return 1

    # first two variables are node-centered coordinates; the rest are cell data
    data = vals[2 * node_count:]
    if data.size != (len(variables) - 2) * cell_count:
        print(f"{file}: unexpected data size ({data.size}, expected "
              f"{(len(variables) - 2) * cell_count})")
        return 1
    channels = [data[i * cell_count:(i + 1) * cell_count].reshape(cell0, cell1).T
                for i in range(len(variables) - 2)]

    if len(channels) < 3:
        print(f"{file}: expected at least 3 data variables, got {len(channels)}")
        return 1
    rgb = np.stack([to_uint8(c) for c in channels[:3]], axis=-1)

    out = file.rsplit('.', 1)[0] + '.png'
    Image.fromarray(rgb, 'RGB').save(out)
    print(f"saved {out} ({rgb.shape[1]}x{rgb.shape[0]})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
