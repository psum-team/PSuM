"""Compare archived runs by identical request block, population, backend, layout and channel."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import t

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def blocks(path):
    return np.genfromtxt(path, names=True, delimiter=',')


def same_requests(a, b):
    for field in ('block', 'channel', 'trials', 'requests', 'capacity_pairs', 'pairs', 'expected_cross', 'dt'):
        np.testing.assert_array_equal(a[field], b[field])


def compare(small, output):
    full = small / 'rate-precision'
    ablation = full / 'ablation'
    rate, pair_counts = [], []
    for variant in ('full', 'no-shift', 'no-feistel', 'no-shift-no-feistel'):
        root = full if variant=='full' else ablation/variant
        for r in read(root/'summary.csv'):
            gpu = r['backend']=='GPU'
            dev = 'cuda' if gpu else 'cpu'
            local = r['per_cell']=='16'
            directory = ('rate-local-' if local else 'rate-') + dev
            prefix = '16x128' if local else '2048'
            path = Path(directory)/f"{prefix}-2-{r['layout']}-blocks.csv"
            a, b = blocks(root/path), blocks(full/path)
            same_requests(a,b)
            c = int(r['channel'])
            ac, bc = a[a['channel']==c], b[b['channel']==c]
            # Paired difference in percentage points of physical expectation.
            delta = 100*(ac['hits']-bc['hits'])/(float(r['expected'])/len(ac))
            # Keep the original 32 comparisons together. The follow-up comparison
            # against the full implementation has its own 16-contrast family.
            contrasts = 16 if variant == 'no-shift-no-feistel' else 32
            radius = t.ppf(1-.05/(2*contrasts),19)*delta.std(ddof=1)/np.sqrt(len(ac))
            rate.append(dict(variant=variant, **r, cross_ratio=ac['cross'].sum()/ac['expected_cross'].sum(),
                             delta_pp=delta.mean(),delta_low=delta.mean()-radius,delta_high=delta.mean()+radius))
        for dev in ('cpu','cuda'):
            ref = small/('factorial-'+dev)
            directory = ref if variant=='full' else root/('pairs-'+dev)
            summary = read(directory/'summary.csv')
            for layout in ('ordered','shuffled'):
                path = f'16-2-{layout}-blocks.csv'
                same_requests(blocks(directory/path),blocks(ref/path))
                rows = sorted((r for r in summary if r['order']==layout),key=lambda r:int(r['channel']))
                assert len(rows)==2 and all(int(r['trials'])==20000 for r in rows)
                pair_counts.append(dict(variant=variant,backend=dev,layout=layout,
                                        outside_c0=rows[0]['pair_outliers'],outside_c1=rows[1]['pair_outliers'],
                                        outside_total=sum(int(r['pair_outliers']) for r in rows)))
    write(output/'rate-comparison.csv',rate)
    write(output/'pair-comparison.csv',pair_counts)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='results directory of the frozen-beam research archive')
    parser.add_argument('--output', type=Path, help='directory for the comparison tables')
    args = parser.parse_args()
    output = args.output or args.results / 'rate-precision' / 'ablation'
    compare(args.results, output)
