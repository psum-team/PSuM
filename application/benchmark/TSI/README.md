# TSI：双流不稳定性（1D） / TSI: two-stream instability (1D)

经典静电双流不稳定性算例：物理设置简单、结果易于检验，用于检验能量守恒与相空间演化。

A classic electrostatic two-stream instability case with a simple setup and easily checked results, used to verify energy conservation and phase-space evolution.

## 对比文献 / Reference

> G. Lapenta,
> "Exactly energy conserving semi-implicit particle in cell formulation",
> *Journal of Computational Physics* **334**, 349–366 (2017).
> DOI: [10.1016/j.jcp.2017.01.002](https://doi.org/10.1016/j.jcp.2017.01.002)

本算例的设置与上述文献一致；注意本算例**不是**文献中的 ECSIM 半隐式格式，仅设置一致。

The setup of this case follows the paper above; note this case is **not** the ECSIM semi-implicit scheme of the paper — only the settings match.

## 运行 / Run

```bash
make tsi
```

> 构建环境需已按根目录文档配置好。
> The build environment is supposed to be already configured per the repository-root docs.

构建并自动运行，输出为 `output/` 下的相空间分布 `.plt` 序列与算例目录下的 `energy.plt` 能量记录。一次运行通常需要几十秒。

Builds and runs automatically. Output is a phase-space distribution `.plt` series under `output/` plus an `energy.plt` energy record in the case directory. A single run usually takes several tens of seconds.

## 结果 / Main results

相空间的双流混合过程与总能量随时间的演化记录。检验要点：总能量漂移应保持在积分轨迹对应的小量级内，电场能量按线性理论增长率发展。

The two-stream mixing in phase space and the recorded total-energy evolution. Check that the total-energy drift stays small relative to the integrated trajectory and that the two-stream structure energy grows at the linear-theory rate.
