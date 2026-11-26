from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import t
root=Path.cwd()
rows=np.genfromtxt(root/'summary.csv',delimiter=',',names=True)
ratio=[]
for row in rows:
    raw=np.loadtxt(root/str(int(row['seed']))/'heat.csv',delimiter=',',skiprows=1,max_rows=1)
    hot,cold=raw[2:4]
    q=1.602176634e-19; m=9.109e-31; ep=8.854187817e-12; n=1e19
    v=np.sqrt(q*(hot+cold)/m)
    b=max(1.054571817e-34/(.5*m*v),q*q/(4*np.pi*ep*v*v*.5*m))
    def inv_rate(gap):
        x=(hot+cold+gap)/2; y=(hot+cold-gap)/2
        ld=np.sqrt(ep/(q*n*(1/x+1/y)))
        tau=3*(4*np.pi*ep)**2*m*m*v**3/(8*np.sqrt(2*np.pi)*n*q**4*np.log(max(ld/b,1.01)))
        return 1e6*tau/(2*gap)
    theory=quad(inv_rate,(hot-cold)/np.e,hot-cold)[0]
    ratio.append(100*row['explained_us']/(row['observed_us']-theory))
ratio=np.asarray(ratio)
ci=t.interval(.95,len(ratio)-1,loc=ratio.mean(),scale=ratio.std(ddof=1)/np.sqrt(len(ratio)))
with (root/'report.md').open('a') as f:
    print(f'\n附加检验：用各轨迹的实际初始温度计算连续时间理论曲线，相应解释比例均值 {ratio.mean():.2f}%，95% 区间 [{ci[0]:.2f}%, {ci[1]:.2f}%]。',file=f)
print(ratio,ratio.mean(),ci)
