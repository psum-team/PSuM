# CCP：容性耦合等离子体（1D） / CCP: capacitively coupled plasma (1D)

一维容性耦合等离子体（CCP）算例：He 放电 + MCC，RF 偏压 450 V / 13.56 MHz。

A 1D capacitively coupled plasma (CCP) case: He discharge with MCC, RF bias 450 V / 13.56 MHz.

## 对比文献 / Reference

> M. M. Turner, A. Derzsi, Z. Donkó, D. Eremin, S. J. Kelly, T. Lafleur, and T. Mussenbrock,
> "Simulation benchmarks for low-pressure plasmas: Capacitive discharges",
> *Physics of Plasmas* **20**, 013507 (2013).
> DOI: [10.1063/1.4775084](https://doi.org/10.1063/1.4775084)

该文为低气压容性放电的多码盲测基准研究：完整规定物理与数值参数，由五个独立开发的 PIC 代码模拟，结果在定义的不确定度范围内统计不可区分，可作为检验其他代码的基准。

The paper is a multi-code blind benchmark study for low-pressure capacitive discharges:
physics and numerical parameters are fully specified, five independently developed PIC
codes participate, and the results are statistically indistinguishable within the
defined uncertainty bands, making them a benchmark for validating other codes.

## 参考数据 / Reference data

`reference_results/` 目录下的 `literature_case1.plt` / `literature_case2.plt` 为上述文献基准算例的参考数据（Tecplot ASCII 格式，`variables` 头标明变量清单），用于本算例的回归比对。

`literature_case1.plt` / `literature_case2.plt` under `reference_results/` are the
reference datasets of the benchmark cases above (Tecplot ASCII format; the variable
list is given in the `variables` header), used for regression comparison of this case.

## 运行 / Run

```bash
make ccp
```

> 构建环境需已按根目录文档配置好。
> The build environment is supposed to be already configured per the repository-root docs.

构建并自动运行，输出为 `output/` 下的 Tecplot ASCII `.plt` 序列；与参考数据叠加的 `comparison.plt` 生成在算例目录下。
通常需要运行几十分钟。

Builds and runs automatically. Output is a Tecplot ASCII `.plt` series under `output/`; `comparison.plt` (overlaid with the reference data) is written to the case directory.
This case usually takes several tens of minutes to run.

## 结果 / Main results

稳态下电子/离子密度的时间平均分布，以及与参考数据的叠加对比图。

Steady-state electron/ion time-averaged density profiles and an overlay comparison against the reference data.