# 400 Td 氩电离 / Argon ionization at 400 Td

单格均匀氩气在恒定电场中电离。两个主程序都用 PSuM DPMCC 处理电子–氩弹性碰撞、电离和 Ar–Ar 弹性碰撞。`main.cpp` 复现 Merzbild 的异权重论文模型，IST-Lisbon 截面表内置于 `argon.hpp`。`main_simple.cpp` 使用 MPDT 截面和碰撞运动学，按权重比移除中性氩。

A uniform argon gas in a single cell ionizes in a constant electric field. Both drivers use PSuM DPMCC for elastic electron–argon scattering, electron-impact ionization and elastic Ar–Ar scattering. `main.cpp` reproduces the unequal-weight model from the Merzbild paper using the IST-Lisbon cross sections embedded in `argon.hpp`. `main_simple.cpp` uses MPDT cross sections and collision kinematics and removes neutral argon with a probability given by the particle weight ratio.

## 构建与运行 / Build and run

在本目录使用 GNU Make、AdaptiveCpp、Clang 16 和 Eigen3。默认 CPU 编译器位于 `/opt/AdaptiveCpp` 和 `/usr/lib/llvm-16`，可通过 `ACPP`、`CPU_FLAGS` 覆盖。

Run the commands below from this directory with GNU Make, AdaptiveCpp, Clang 16 and Eigen3. The default CPU build uses `/opt/AdaptiveCpp` and `/usr/lib/llvm-16`. Override the compiler and flags with `ACPP` and `CPU_FLAGS`.

```sh
make                                  # 构建两个主程序 / build both drivers
make run STEPS=1000 START=250         # 论文模型短算例 / short run of the published model
make run-simple STEPS=1000 START=250  # MPDT 模型短算例 / short run of the MPDT model
```

完整算例运行 `make run` 或 `make run-simple`。默认 500000 步，从第 150000 步开始统计，电子目标 6000，种子 1234。配置好 NVIDIA 后端后，`make run-gpu` 运行论文模型。时间历程、电子能谱和日志写入 `results/`，可用 `OUT` 指定 CSV 路径。

Use `make run` or `make run-simple` for a full run. Both default to 500000 steps, with statistics starting at step 150000, a target of 6000 electron macroparticles and seed 1234. With the NVIDIA backend configured, `make run-gpu` runs the published model. Time histories, electron energy spectra and logs go to `results/`. Set `OUT` to override the CSV path.

## 参考结果 / Reference results

[论文 AR 和 ES 参考值](reference_results/reference.json) · [400 Td 三模型与中性分辨率对照](reference_results/full400-summary.json) · [Merzbild 论文](https://doi.org/10.1016/j.jcp.2026.115503)

[Published AR and ES values](reference_results/reference.json) · [Three-model and neutral-resolution comparison at 400 Td](reference_results/full400-summary.json) · [Merzbild paper](https://doi.org/10.1016/j.jcp.2026.115503)
