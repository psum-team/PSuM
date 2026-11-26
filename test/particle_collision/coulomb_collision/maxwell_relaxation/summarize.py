from pathlib import Path
import csv,re
import numpy as np
from scipy.stats import t
root=Path.cwd()
rows=[]
fingerprints=set()
for seed in range(301,307):
    d=root/str(seed)
    fingerprints.update(line.split()[0] for line in (d/'inputs.sha256').read_text().splitlines())
    a=(d/'analysis.txt').read_text()
    obs,theory=map(float,re.search('模拟和论文理论输出插值： (.*)',a)[1].split())
    real,mw,calc=map(float,re.search('真实、Maxwell、calc_tau： (.*)',a)[1].split())
    rates=np.loadtxt(d/'rates.csv',skiprows=1,delimiter=',')
    assert rates.shape==(53,10)
    excess=obs-theory
    rows.append([seed,obs,theory,real,mw,excess,real-mw,100*(real-mw)/excess,excess-real+mw])
assert len(fingerprints)==12
headers=['seed','observed_us','theory_us','real_flux_us','maxwell_flux_us','extra_us','explained_us','explained_percent','residual_us']
with (root/'summary.csv').open('w') as f:
    w=csv.writer(f); w.writerow(headers); w.writerows(rows)
v=np.array(rows)[:,1:]
def stats(x):
    m=x.mean(); hw=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return m,m-hw,m+hw
with (root/'report.md').open('w') as f:
    print('# 双电子温差弛豫，六组独立 GPU 初态\n', file=f)
    print('两群电子各 100000 个，初始标称温度 300/150 eV，各群密度 1e19 m⁻³；每步 0.2 ns，共 200000 步。每条轨迹取 53 个时刻，真实速度和同温 Maxwell 重采样用同一 Nanbu 热流诊断。轨迹、配对和热流诊断分别用独立种子生成。\n',file=f)
    print('解释比例 = (真实速度热流重建时间 − Maxwell 热流重建时间) / (模拟实测时间 − 理论时间)。每条曲线以自身初始温差的 1/e 为终点。重建时间沿实测温差积分。区间为按独立轨迹计算的双侧 95% Student t 区间，自由度 5。上轮试验单列，不参与估计。\n',file=f)
    print('| seed | 实测 µs | 理论 µs | 真实热流 µs | Maxwell 热流 µs | 额外 µs | 解释 µs | 解释 % | 剩余 µs |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|',file=f)
    for row in rows: print('|'+str(row[0])+'|'+ '|'.join(f'{x:.4f}' for x in row[1:])+'|',file=f)
    print('\n|量|均值|95% 下限|95% 上限|\n|---|---:|---:|---:|',file=f)
    for name,x in zip(headers[1:],v.T):
        print('|'+name+'|'+'|'.join(f'{n:.5f}' for n in stats(x))+'|',file=f)
    print('\n上轮独立对照：实测 7.549 µs，解释比例 97.9%。',file=f)
print((root/'report.md').read_text())
