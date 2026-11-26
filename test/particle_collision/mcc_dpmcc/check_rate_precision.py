"""Check the production frozen-beam rate for both backends (ROOT SUMMARY_CSV)."""
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t


def check(root, output):
    # Independent physical reference: (N/2)^2 counter-streaming pairs per cell,
    # each with macro-weight density/N and half the total N2 VHS cross section.
    speed, density = 1000., 1e21
    sigma = (math.pi * (4.17e-10)**2
             * (2 * 1.380649e-23 * 273 / (.5 * 4.652e-26 * speed**2))**.24
             / math.gamma(1.76))
    rows = []
    for n,cells,prefix,folder in ((2048,1,'2048','rate-'), (16,128,'16x128','rate-local-')):
        for backend, suffix in (('CPU', 'cpu'), ('GPU', 'cuda')):
            directory = root / (folder+suffix)
            for layout in ('ordered', 'shuffled'):
                blocks = np.genfromtxt(directory / f'{prefix}-2-{layout}-blocks.csv', names=True, delimiter=',')
                for c in (0, 1):
                    b = blocks[blocks['channel'] == c]
                    if not (np.array_equal(b['block'], np.arange(20)) and np.all(b['trials'] == 1000)
                            and np.all(b['dt'] == 1e-8) and np.all(b['capacity_pairs'] == b['pairs'])
                            and np.all(b['hits'] <= b['cross']) and np.all(b['cross'] <= b['pairs'])):
                        raise ValueError(f'Incomplete or inconsistent block data: {backend} {layout} {c}')
                    expected = cells * (n/2)**2 * density/n * sigma/2 * speed * b['dt'] * b['trials']
                    ratios = b['hits'] / expected
                    # Simultaneous 95% intervals across 2 populations x 2 backends x 2 layouts x 2 channels.
                    radius = t.ppf(1-.05/(2*16), 19) * ratios.std(ddof=1)/math.sqrt(20)
                    mean = ratios.mean()
                    rows.append(dict(per_cell=n,cells=cells,backend=backend,layout=layout,channel=c,
                                     requests=int(sum(b['requests'])),matched=int(sum(b['pairs'])),hits=int(sum(b['hits'])),
                                     expected=sum(expected),ratio=mean,low=mean-radius,high=mean+radius,
                                     passed=mean-radius>=.95 and mean+radius<=1.05))
    with output.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return all(r['passed'] for r in rows)


if __name__ == '__main__':
    root, output = map(Path, sys.argv[1:])
    sys.exit(0 if check(root, output) else 1)
