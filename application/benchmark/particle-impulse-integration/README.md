# PII：粒子脉冲积分 / PII: particle impulse integration

自由分子流稳态快速算法（粒子脉冲积分）算例。

A steady-state free-molecular-flow case computed with the fast particle impulse integration algorithm.

## 对比文献 / Reference

PII 算例没有独立的对比文献；**该算例本身即以下论文所使用的 case**.

The PII case has no separate comparison literature; the case itself is the one used in
the following paper.

> R. Mao, J. Ren, Z. Wang, W. Luo, Y. Wang, Z. Li, and H. Tang,
> "High-efficiency rarefied neutral gas computational model for plasma simulation",
> *Physical Review E* **112**, 055310 (2025).
> DOI: [10.1103/8znn-h985](https://doi.org/10.1103/8znn-h985)

## 参考数据 / Reference data

`reference_results/hp_density_pii.plt` 为本算例的参考数据（Tecplot ASCII 格式），用于回归比对。

`reference_results/hp_density_pii.plt` is the reference dataset of this case (Tecplot ASCII format), used for regression comparison.

## 运行 / Run

```bash
make pii
```

> 构建环境需已按根目录文档配置好。
> The build environment is supposed to be already configured per the repository-root docs.

构建、运行并自动清理二进制，输出 `density_pii.plt`（Tecplot ASCII）。工作站上通常为分钟量级。

Builds, runs, and removes the binary automatically. Output: `density_pii.plt` (Tecplot ASCII). Typically minutes on a workstation.

## 结果 / Main results

羽流区域的稳态密度分布；与参考数据对比，羽流区密度应在百分之几量级内一致。

The steady-state density profile in the plume region; compared with the reference data, plume-region densities should agree within a few percent.
