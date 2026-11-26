from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student
raw=np.loadtxt('heat.csv',skiprows=1,delimiter=',').reshape(-1,24,8)
a=np.loadtxt('coulomb_doubleT_validation.plt');assert a.shape==(1001,8) and np.isfinite(raw).all() and raw.shape==(53,24,8)
ts=raw[:,0,0]*.0002;t=a[:,1]*1e6;G=raw[:,0,2]-raw[:,0,3]; hot,cold=raw[:,0,2:4].T
H=raw[:,:,4:6];h=H.mean(1);se=H.std(1,ddof=1)/np.sqrt(24)
assert np.array_equal(raw[:,:,1],np.tile(np.arange(24),(53,1))) and np.all(np.diff(ts)>0)
def theory(x,y):
 q=1.602176634e-19;m=9.109e-31;ep=8.854187817e-12;n=1e19
 v=np.sqrt(q*(x+y)/m);mu=m/2;ld=np.sqrt(ep/(q*n*(1/x+1/y)))
 b=np.maximum(1.054571817e-34/(mu*v),q*q/(4*np.pi*ep*v*v*mu))
 tau=3*(4*np.pi*ep)**2*m*m*v**3/(8*np.sqrt(2*np.pi)*n*q**4*np.log(np.maximum(ld/b,1.01)))
 return 2*(x-y)/tau*1e-6
def crossing(tt,gg):
 threshold=gg[0]/np.e;j=np.flatnonzero(gg<=threshold)[0];return np.interp(threshold,gg[j-1:j+1][::-1],tt[j-1:j+1][::-1])
g=a[:,2]-a[:,3];gth=a[:,4]-a[:,5];tobs=crossing(t,g);tth=crossing(t,gth);threshold=G[0]/np.e
k=np.flatnonzero(G<=threshold)[0]
assert np.all(np.diff(G[:k+1])<0)
frac=(G[k-1]-threshold)/(G[k-1]-G[k])
gap=np.r_[G[:k],threshold]
def duration(flux):
 vals=np.concatenate((flux[:k],(flux[k-1]+frac*(flux[k]-flux[k-1]))[None,...]));assert np.min(vals)>0
 return np.einsum('i,i...->...',-np.diff(gap),.5*(1/vals[1:]+1/vals[:-1]))
tr,tm=duration(h)
# Resample independent diagnostic batches within each snapshot, keeping pairing of real/Maxwell estimates.
rng=np.random.default_rng(823);boot=[]
for rep in range(1000):
 ids=rng.integers(24,size=(k+1,24))
 boot.append(duration(H[np.arange(k+1)[:,None],ids].mean(1)))
boot=np.array(boot)
expl=(tr-tm)/(tobs-tth);bounds=np.percentile((boot[:,0]-boot[:,1])/(tobs-tth),[2.5,97.5])
calc=theory(hot,cold)
def slope(x):
 lo=max(0,min(39,x-.5));hi=lo+1
 ix=(t>=lo-1e-10)&(t<=hi+1e-10)
 return -np.polyfit(t[ix],g[ix],1)[0]
actual=np.array([slope(x) for x in ts])
np.savetxt('rates.csv',np.column_stack((ts,G,h,se,calc,actual,raw[:,0,6:8])),delimiter=',',header='time_us,gap_eV,real_eV_us,maxwell_eV_us,SE_real,SE_maxwell,calc_tau_eV_us,trajectory_local_eV_us,hot_excess4,cold_excess4',comments='')
print('# GPU 双电子弛豫的冻结热流对照\n')
print('单条新版 20 万步轨迹，两群各 10 万粒子；沿用共同初态。0–16 µs 每 0.4 µs 取快照，随后每 2 µs，合计 53 张。每个时刻以 24 个独立批次抽取跨群对，每批 524288 对；真实与重采样共用配对索引和相对横向方位角。保留真实 vx 和 v⊥²，方位角按原降维算法在诊断中重建。每批 Maxwell 独立抽取两群各 10 万粒子，按组缩放以精确匹配快照的原始二阶矩温度。核的速率使用正式 s、截面上限和查表，解析平均散射角；乘均匀混池的跨标签概率。所有热流为热冷两群温差减少率 (eV/µs)，单群冷却率为其一半。calc_tau 用快照两温度求值，实际热流以邻近 1 µs 窗口线性拟合，首末端为单侧窗口。')
print('\n|µs|真实|Maxwell|calc_tau|轨迹局部|真实/Maxwell|热组超额四阶矩|冷组超额四阶矩|\n|---:|---:|---:|---:|---:|---:|---:|---:|')
for x in [0,.4,1.2,2,4,6,7.6,8,12,16,20,30,40]:
 i=np.argmin(abs(ts-x));print(f'|{ts[i]:g}|'+ '|'.join(f'{z:.3f}' for z in [*h[i],calc[i],actual[i],h[i,0]/h[i,1],*raw[i,0,6:8]])+'|')
print('\n到相同温差阈值 ΔT₀/e 的时间（µs）')
print('模拟和论文理论输出插值：',tobs,tth)
print('沿实测降温路径以相同温差节点积分 d(ΔT)/热流，真实、Maxwell、calc_tau：',tr,tm,duration(calc))
print('积分热流时间的诊断抽样区间（真实、Maxwell）:',np.percentile(boot,[2.5,97.5],axis=0))
print('相对观测额外耗时的解释比例及诊断抽样区间：',expl,bounds)
print('用每隔一张快照重积分的敏感性：')
for col in h.T:
 ixs=np.arange(0,k,2);xx=np.r_[G[ixs],threshold];yy=np.r_[col[ixs],col[k-1]+frac*(col[k]-col[k-1])];print(np.sum(-np.diff(xx)*(1/yy[:-1]+1/yy[1:])/2))
print('分布超额四阶矩取 3〈v⁴〉/(5〈v²〉²)−1。抽样误差按快照内批次计算，未包含轨迹间差异。')
fig,ax=plt.subplots(2,1,figsize=(8,7),sharex=True)
for j,n in enumerate(['Snapshot velocities','Matched Maxwellian']):
 ax[0].plot(ts,h[:,j],label=n);ax[0].fill_between(ts,h[:,j]-2*se[:,j],h[:,j]+2*se[:,j],alpha=.15)
ax[0].plot(ts,calc,'k--',label='calc_tau');ax[0].plot(ts,actual,'.',label='Trajectory (1 µs window)');ax[0].set_ylabel('Gap loss (eV/µs)');ax[0].legend()
ax[1].plot(t,g,label='Trajectory');ax[1].plot(t,gth,'--',label='Paper theory');ax[1].axhline(threshold,color='0.6',ls=':');ax[1].set(xlabel='Time (µs)',ylabel='Temperature gap (eV)');ax[1].legend();fig.tight_layout();fig.savefig('comparison.png',dpi=150)
