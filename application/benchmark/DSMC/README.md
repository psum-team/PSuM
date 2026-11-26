# DSMC 氮气超声速平板绕流 / DSMC of supersonic nitrogen flow past a flat plate

`main.cpp` 使用 VHS 碰撞模型和转动内能交换计算稀薄氮气绕平板的流动。工况对应 Palharini 等人的 truncated flat plates 算例。[原始论文](https://doi.org/10.1016/j.compfluid.2015.07.021)

`main.cpp` simulates rarefied nitrogen flow past a flat plate using variable hard sphere (VHS) collisions and rotational energy exchange. The conditions follow the truncated flat plate case of Palharini et al. [Original paper](https://doi.org/10.1016/j.compfluid.2015.07.021)

## 构建与运行 / Build and run

在本目录使用 GNU Make、AdaptiveCpp、Clang 16 和 Eigen3。可用 `ACPP` 和 `CPU_CXX` 指定编译器路径。

Use GNU Make, AdaptiveCpp, Clang 16 and Eigen3 from this directory. Set `ACPP` and `CPU_CXX` to override the compiler paths.

```sh
make                         # 构建默认 CPU 主程序 / build the default CPU driver
OMP_NUM_THREADS=2 make smoke # 同一主程序，运行 100 步 / run 100 steps with the same driver
make run                     # 默认 50000 步 / run the default 50000 steps
```

默认入口种子 20261021，宏粒子权重 10¹⁰，时间步长 2×10⁻⁸ s。诊断写入 `output/local-moments.csv`，每五步记录七个测区。重复运行会覆盖同名文件。设置 `N2_DIAGNOSTICS=0` 可关闭诊断。配置好 CUDA 后运行 `make run-gpu`，开启诊断时另写 `gpu-memory.csv`。

The defaults are inlet seed 20261021, macroparticle weight 10¹⁰ and time step 2×10⁻⁸ s. Diagnostics record seven sampling regions every five steps in `output/local-moments.csv`. Rerunning overwrites files with the same name. Set `N2_DIAGNOSTICS=0` to disable diagnostics. With CUDA configured, use `make run-gpu`. GPU runs also write `gpu-memory.csv` when diagnostics are enabled.

## 参考结果 / Reference results

[历史对照工况](reference_results/reference-case.json) · [七测区数密度](reference_results/steady-density.csv) · [平动与转动温度](reference_results/steady-temperature.csv)。两张表的 SPARTA 列来自权重 5×10¹⁰、三个种子、1–2 ms 时间窗的历史对照。

[Historical reference case](reference_results/reference-case.json) · [Number densities in the seven sampling regions](reference_results/steady-density.csv) · [Translational and rotational temperatures](reference_results/steady-temperature.csv). The SPARTA columns in both tables come from the historical comparison with macroparticle weight 5×10¹⁰, three seeds and a 1–2 ms sampling window.
