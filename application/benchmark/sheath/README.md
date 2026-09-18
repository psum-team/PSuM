# sheath：等离子体鞘层（1D） / sheath: plasma sheath (1D)

一维等离子体鞘层算例，偏压壁面 12.5 V。

A 1D plasma sheath case with a biased wall at 12.5 V.

## 对比文献 / Reference

> G. R. Werner, T. G. Jenkins, A. M. Chap, and J. R. Cary,
> "Speeding up simulations by slowing down particles: Speed-limited particle-in-cell simulation",
> *Physics of Plasmas* **25**, 123512 (2018).
> DOI: [10.1063/1.5061683](https://doi.org/10.1063/1.5061683)

本算例以上述文献为对比参考。注意本算例**不是**文献中的 SLPIC 格式，仅设置一致。

This case uses the paper above as its comparison reference. Note this case is **not** the SLPIC scheme of the paper — only the settings match.

## 运行 / Run

```bash
make sheath
```

> 构建环境需已按根目录文档配置好。
> The build environment is supposed to be already configured per the repository-root docs.

构建并自动运行，输出为 `output/` 下的 `.plt` 序列。一次运行通常需要几十秒。

Builds and runs automatically. Output is a `.plt` series under `output/`. A single run usually takes several tens of seconds.

## 结果 / Main results

到达稳态后的鞘层电位与密度剖面；可与文献中的一维分布做比较。

Steady-state sheath potential and density profiles; compare with the profiles in the reference.
