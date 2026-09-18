# EDI：ExB 漂移放电（2D） / EDI: 2D ExB drift discharge

模拟霍尔推进器通道内的等离子体不稳定性关联的输运过程，含磁场剖面与电离源项。

Simulates the instability-related transport processes of the plasma in a Hall-thruster channel, with a magnetic field profile and ionization source terms.

## 对比文献 / Reference

> T. Charoy, J. P. Boeuf, A. Bourdon, J. A. Carlsson, P. Chabert, B. Cuenot, D. Eremin,
> L. Garrigues, K. Hara, I. D. Kaganovich, A. T. Powis, A. Smolyakov, D. Sydorenko,
> A. Tavant, O. Vermorel, and W. Villafana,
> "2D axial-azimuthal particle-in-cell benchmark for low-temperature partially magnetized plasmas",
> *Plasma Sources Science and Technology* **28**, 105010 (2019).
> DOI: [10.1088/1361-6595/ab46c5](https://doi.org/10.1088/1361-6595/ab46c5)

本算例以上述多码基准研究为对比文献。

This case uses the multi-code benchmark study above as its comparison reference.

## 运行 / Run

```bash
make edi    # 构建并创建 output/
./edi       # 必须在 EDI 根目录运行（写出 output/... 相对路径）
```

> 构建环境需已按根目录文档配置好。
> The build environment is supposed to be already configured per the repository-root docs.

每 5,000 步输出一帧（场量 `.plt` 与耗时记录），总步数为 6e6（`EDI.cpp`），约400万步时收敛。
单 4090 运行需要30小时以上；快速验证请先调小循环上限。

One frame (field `.plt` plus a timing record) every 5,000 steps. This is a long GPU benchmark test: the total number of steps is fixed at 6e6, and it basically converges at 4e6. It took more than 30 hours on the NVIDIA 4090 GPU. For a quick check, reduce the loop bound first.

## 结果 / Main results

通道等离子体的密度/电场等随时间的演化序列；可与文献中密度与电位剖面的量级和位置进行对照检验。

Time evolution of the channel-plasma density, electric field, and other distributions; check against the magnitude and location of the density and potential profiles in the reference.
