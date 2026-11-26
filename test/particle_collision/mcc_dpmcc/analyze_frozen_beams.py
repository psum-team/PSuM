"""Summarize test_frozen_beams --frozen-rates DIR. Pair rates use *all* trials as denominator."""
import csv
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binom, t


def mean_interval(values):
    mean = np.mean(values)
    half = t.ppf(.975, len(values)-1) * np.std(values, ddof=1) / math.sqrt(len(values))
    return mean, mean-half, mean+half


def request_draws(n, size, block):
    # Scalar-seeded MT19937 matches std::mt19937; advance the production
    # particle RNG once to replay the request draw by stable ID.
    seeds = np.random.RandomState(19937 + block).randint(0, 2**32, (size,n), dtype=np.uint32)
    return (seeds * np.uint32(1664525) + np.uint32(1013904223)) / (2**32-1)


def expected_neighbors(n, channels, blocks, width):
    """Uniform partner baseline for circular ID distance <= width, given roles."""
    expected = np.zeros(channels)
    for block in range(20):
        size = int(blocks['trials'][blocks['block'] == block][0])
        draws = request_draws(n, size, block)
        bg = draws >= .1
        nb = bg.sum(axis=1)
        left = nb.copy()
        padded = np.pad(bg, ((0,0), (width,width)), mode='wrap')
        prefix = np.pad(padded.cumsum(axis=1), ((0,0),(1,0)))
        near = prefix[:,2*width+1:] - prefix[:,:-(2*width+1)]
        for c in range(channels):
            inc = (draws >= c*.1/channels) & (draws < (c+1)*.1/channels)
            m = inc.sum(axis=1)
            k = np.minimum(m, left)
            left -= k
            row = blocks[(blocks['channel']==c) & (blocks['block']==block)][0]
            assert m.sum() == row['requests'] and k.sum() == row['capacity_pairs']
            expected[c] += ((inc*near).sum(axis=1)*k / np.maximum(1, m*nb)).sum()
    return expected


def conditional_pairs(directory, prefix, n, channels, blocks, pairs):
    """Replay the small cells by stable ID and condition on both actual role sets."""
    trials = int(sum(blocks['trials'][blocks['channel'] == 0]))
    expected = np.zeros((channels, n, n))
    variance = np.zeros_like(expected)
    for block in range(20):
        size = trials // 20
        draws = request_draws(n, size, block)
        bg = draws >= .1
        nb = bg.sum(axis=1)
        left = nb.copy()
        for c in range(channels):
            inc = (draws >= c*.1/channels) & (draws < (c+1)*.1/channels)
            m = inc.sum(axis=1)
            k = np.minimum(m, left)
            left -= k
            row = blocks[(blocks['channel']==c) & (blocks['block']==block)][0]
            assert m.sum() == row['requests'] and k.sum() == row['capacity_pairs']
            scale = k / np.maximum(1, m*nb)
            directed = inc[:,:,None] & bg[:,None,:]
            prob = (directed | directed.transpose(0,2,1)) * scale[:,None,None]
            assert np.isclose(prob[:, :n//2, n//2:].sum(), row['expected_cross'])
            expected[c] += prob.sum(axis=0)
            variance[c] += (prob*(1-prob)).sum(axis=0)
    observed = np.zeros_like(expected, dtype=int)
    hits = np.zeros_like(observed)
    index = tuple(pairs[key] for key in ('channel','i','j'))
    observed[index] = pairs['pairs']
    hits[index] = pairs['hits']
    with (directory / (prefix+'-conditional.csv')).open('w') as f:
        writer = csv.writer(f)
        writer.writerow(('i','j','channel','pair_rate','uniform_pair_rate_given_roles','collision_rate','pair_residual_sd'))
        for i in range(n):
            for j in range(i+1,n):
                for c in range(channels):
                    idx = (c,i,j)
                    sd = math.sqrt(variance[idx])
                    residual = (observed[idx]-expected[idx])/sd if sd else 0.
                    writer.writerow((i,j,c,observed[idx]/trials,expected[idx]/trials,hits[idx]/trials,residual))
    return expected, variance


def small_cell_pair_reference(n, channels, c):
    # Average the channel-priority capacity over the multinomial role assignment.
    expected = 0.
    for m in range(n+1):
        for first in range(m+1):
            p = binom.pmf(m, n, .1) * binom.pmf(first, m, 1/channels)
            first_matches = min(first, n-m)
            k = first_matches if c == 0 else min(m-first, n-m-first_matches)
            expected += p * k
    return expected / math.comb(n, 2)


def summarize_controls(directory, prefix, expected, variance, native, trials):
    path = directory / (prefix+'-controls.csv')
    if not path.exists():
        return []
    records = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    summary = []
    raw = {(r['channel'],r['i'],r['j']): (r['pairs'],r['hits']) for r in native}
    for mode in np.unique(records['mode']):
        for c in range(2):
            rows = records[(records['mode']==mode) & (records['channel']==c)]
            idx = (c, rows['i'], rows['j'])
            assert len(rows)==120 and np.all(rows['trials']==trials)
            assert rows['pairs'].sum()==sum(v[0] for k,v in raw.items() if k[0]==c)
            if mode=='native':
                assert all(raw.get((c,r['i'],r['j']), (0,0))==(r['pairs'],r['hits']) for r in rows)
            z = (rows['pairs']-expected[idx])/np.sqrt(variance[idx])
            # Same simultaneous uniform-pair test as the original diagnostic.
            tail = .05/(2*240)
            lo, hi = binom.ppf([tail,1-tail],trials,small_cell_pair_reference(16,2,c))
            summary.append(dict(layout=prefix.split('-')[-1],mode=mode,channel=c,
                pairs=int(rows['pairs'].sum()),hits=int(rows['hits'].sum()),
                rms_z=np.sqrt(np.mean(z*z)),max_abs_z=np.abs(z).max(),
                outside=int(((rows['pairs']<lo)|(rows['pairs']>hi)).sum())))
    return summary


def summarize(directory):
    summary = []
    controls = []
    for path in sorted(directory.glob('*-blocks.csv')):
        size, channels, order, _ = path.name.split('-')
        n, channels = int(size), int(channels)
        blocks = np.genfromtxt(path, delimiter=',', names=True)
        pairs_path = path.with_name(path.name.replace('blocks', 'pairs'))
        pairs = np.atleast_1d(np.loadtxt(pairs_path, delimiter=',', skiprows=1, dtype=[(key, 'i8') for key in ('i','j','channel','trials','pairs','hits')]))
        assert len(blocks) == 20 * channels
        if n <= 16:
            prefix = path.name.removesuffix("-blocks.csv")
            expected, variance = conditional_pairs(directory, prefix, n, channels, blocks, pairs)
            controls.extend(summarize_controls(directory, prefix, expected, variance, pairs, int(blocks['trials'][0])*20))
        if n == 2048:
            width = n//64
            neighbors = expected_neighbors(n, channels, blocks, width)
        for c in range(channels):
            b = blocks[blocks['channel'] == c]
            trials = sum(b['trials'])
            rows = pairs[pairs['channel'] == c]
            assert rows['pairs'].sum() == sum(b['pairs'])
            assert rows['hits'].sum() == sum(b['hits'])
            # Independent equal-weight, unordered-pair physical target, unit volume.
            sigma = math.pi * (4.17e-10)**2 * (2*1.380649e-23*273/(0.5*4.652e-26*1000**2))**.24 / math.gamma(1.76)
            target = (n//2)**2 * 1e21/n * sigma*1000*b['dt']/channels
            cross_acceptance = 0.5*(1-1/n)*sigma*1000*1e21*b['dt']/.1
            physical = b['hits']/b['trials']/target
            mean, low, high = mean_interval(physical)
            # Residual against a uniform assignment conditional on the realized requests.
            cross_ratio = sum(b['cross']) / sum(b['expected_cross'])
            _, residual_lo, residual_hi = mean_interval((b['cross']-b['expected_cross']) / np.mean(b['expected_cross']))
            cross_rows = rows[(rows['i'] < n//2) & (rows['j'] >= n//2)]
            # Include never-observed pairs in the range and mean.
            total_pairs = n*(n-1)//2
            neighbor_ratio = None
            if n == 2048:
                separation = rows['j'] - rows['i']
                observed = rows['pairs'][np.minimum(separation, n-separation) <= width].sum()
                neighbor_ratio = observed / neighbors[c]
            outliers = None
            if n <= 16:
                rate = small_cell_pair_reference(n, channels, c)
                # Each fixed pair is observed at most once per trial. Bonferroni
                # covers all pairs and both channels of this configuration.
                tail = .05 / (2 * channels * total_pairs)
                low_count, high_count = binom.ppf([tail, 1-tail], int(trials), rate)
                outliers = int(np.sum((rows['pairs'] < low_count) | (rows['pairs'] > high_count)))
                if low_count > 0:
                    outliers += total_pairs-len(rows)
            summary.append(dict(n=n,channels=channels,order=order,channel=c,trials=int(trials),
                requests=int(sum(b['requests'])), matched=int(sum(b['pairs'])), accepted=int(sum(b['hits'])),
                pair_outliers=outliers,near_id_uniform_ratio=neighbor_ratio,role_fill=sum(b['pairs'])/sum(b['requests']),cross_uniform_ratio=cross_ratio,
                cross_lo=1+residual_lo,cross_hi=1+residual_hi,
                accept_cond_ratio=sum(b['hits'])/sum(b['cross']*cross_acceptance),
                physical_ratio=mean,physical_lo=low,physical_hi=high,
                mean_pair_rate=rows['pairs'].sum()/trials/total_pairs,
                min_pair_rate=(0 if len(rows)<total_pairs else min(rows['pairs'])/trials),
                max_pair_rate=max(rows['pairs'])/trials,
                mean_cross_hit_rate=cross_rows['hits'].sum()/trials/(n//2)**2))
    if controls:
        with (directory/'pair-order-summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(controls[0]))
            writer.writeheader()
            writer.writerows(controls)
    with (directory/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == '__main__':
    summarize(Path(sys.argv[1]))
