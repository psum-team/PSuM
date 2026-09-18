# DSMC：N₂ 超声速射流撞击平板 / DSMC: supersonic N₂ jet impinging on a flat plate

稀薄气体 DSMC 算例：N₂ 超声速射流撞击平板，VHS 碰撞模型 + 内能交换。

A rarefied-gas DSMC case: a supersonic N₂ jet impinging on a flat plate, with a VHS collision model and internal energy exchange.

## 对比文献 / Reference

> R. C. Palharini, C. White, T. J. Scanlon, R. E. Brown, M. K. Borg, and J. M. Reese,
> "Benchmark numerical simulations of rarefied non-reacting gas flows using an open-source DSMC code",
> *Computers & Fluids* **120**, 140–157 (2015).
> DOI: [10.1016/j.compfluid.2015.07.021](https://doi.org/10.1016/j.compfluid.2015.07.021)

## 对应关系 / Correspondence

本算例对应上述文献中的 **truncated flat plates** 算例。

This case corresponds to the **truncated flat plates** case in the paper above.

使用前请对照原文核对工况参数的一致性，并按需引用上述文献。

Before use, check the flow conditions against the original paper and cite it where appropriate.

## 运行 / Run

```bash
make dsmc
```

> 构建环境需已按根目录文档配置好。
> The build environment is supposed to be already configured per the repository-root docs.

构建并自动运行；输出为运行目录与 `output/` 下的 `.plt` 序列。
通常需要运行几十分钟。

Builds and runs automatically. Output is a `.plt` series in the working directory and `output/` (flow fields and `vx_dist_*.plt` velocity distributions).
This case usually takes several tens of minutes to run.

## 结果 / Main results

平板绕流的流场参数分布的时间演化。可以获得与参考文献基本一致的密度分布。

Flow-field distributions around the plate and the time evolution of the vx velocity distribution.
A density distribution obtained is basically consistent with the reference literature.