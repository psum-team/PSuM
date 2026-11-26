# 库仑碰撞 / Coulomb collisions

`make`（test9）运行 [test_coulomb_collision.cpp](test_coulomb_collision.cpp)。分别测试电子–电子和电子–离子散射，核对粒子总数、物种间能量交换及总动能和总动量守恒。

`make` (target `test9`) runs electron–electron and electron–ion scattering tests. They check particle counts, energy exchange between species, and conservation of total kinetic energy and momentum. [Test source](test_coulomb_collision.cpp)

另有两个边界测试。一个分别在全部和部分有效粒子位于网格外时推进，检查网格外粒子的速度与随机种子。另一个按电子与二价离子在各格的粒子数和温度计算解析德拜长度，改变电子温度后再次核对，并检查空格的结果为 NaN。

Two edge cases are covered. The first advances groups with all or some valid particles outside the grid and checks the velocities and random seeds of the out-of-grid particles. The second computes the analytical Debye length from electron and doubly charged ion counts and temperatures in each cell. It repeats the comparison after changing the electron temperature and checks that empty cells return NaN.

构造时为每个物种绑定粒子组、角色、质量（kg）、带符号电荷（C）及格心温度场（eV）。粒子组和温度场的生命周期需覆盖求解器使用期间。每步更新温度场后直接推进，求解器会读取最新场值。

At construction, bind each species to its particle group, role, mass (kg), signed charge (C) and cell-centered temperature field (eV). The groups and fields must remain alive while the solver uses them. Update the temperature fields before each step and call `solve` to use the latest values.

```cpp
using Role = psum::particle_collision::coulomb_role;
Solver solver(q, grid, {{electrons, Role::electron, me, -e, Te},
                        {ions, Role::ion, mi, 2 * e, Ti}});
solver.solve(dt, num2dens, read_velocity, write_velocity);
```

池共享等权宏粒子的配对机会，每个物种仍独立贡献密度和德拜长度。各池先对全部物种的槽位做一次整组置换再分箱，只移动索引。

Macroparticles of equal weight share pairing opportunities in a pool. Each species contributes separately to density and Debye length. Before binning, each pool permutes the combined slots of all its species. Only indices are moved.

每步依次执行离子池内部、电子池内部、电子与离子三轮。池内偶数粒子两两配对，奇数且至少三个时，前三个两两走半步，其余走整步。跨池每格配 min(电子数, 离子数) 对。电子池和离子池在池内配对时均取本格本池总密度乘以 (N−1)/N，N 为本格实际入格的本池粒子数。电子与离子跨池配对时取两池总密度的较大值。每对按真实物种的质量、电荷和密度上限调用散射核。

Each step runs ion–ion, electron–electron and electron–ion collisions in that order. Within a pool, an even number of particles is paired off. For an odd count of at least three, the first three form three pairs, each advanced by a half step, and the remainder form full-step pairs. Across pools, each cell forms min(electron count, ion count) pairs. Within either pool, scattering uses the cell’s total pool density multiplied by (N−1)/N, where N is the actual number of particles binned into that pool in the cell. Electron–ion scattering uses the larger of the two pool densities. Each pair uses the actual species masses and charges and the density-dependent cross-section cap.

## GPU 回归覆盖 / GPU regression coverage

- `test9` 还在 16 384 个独立格中各放一对等质量反向离子，用正式 Nanbu 核推进。相对速度大小固定，轴向相对速度的系综均值满足 $g_x(k)/g_x(0)=\exp(-ks)$。检查第 1、4、8 步，$s=0.2$，容差为初始相对速度的 0.025。
- 运行 `make test_coulomb_pool` 可检查三物种真实端点、质量、电荷、有效密度和截面上限，空池、单粒子、奇数半步及跨池数目不等的情况。对 4+12 和 8+8 布局检查跨标签比例及每个搭档的频率。比例容差含理论值的 1% 和有限样本余量；该检验针对配对统计。

- `test9` also advances one pair of equal-mass, counterstreaming ions in each of 16 384 independent cells with the production Nanbu kernel. The relative speed has fixed magnitude. The ensemble mean of its axial component follows $g_x(k)/g_x(0)=\exp(-ks)$. The test checks steps 1, 4 and 8 with $s=0.2$ and a tolerance of 0.025 times the initial relative speed.
- `make test_coulomb_pool` checks the endpoints, masses, charges, effective densities and cross-section caps for three species. It covers empty pools, single particles, odd-count half steps and unequal pool sizes. The 4+12 and 8+8 layouts check the cross-label fraction and each partner’s frequency. The fraction tolerance includes 1% of the theoretical value plus a finite-sample allowance.

- `make test_coulomb_relaxation` 对照原有的双电子温度与 Spitzer 电子–离子温度理论演化。短程模式每组 32 768 粒子、512 步，以各格平均实测初温起算，每 128 步检查两组温度的相对误差不超过 2%。`make test_coulomb_relaxation COULOMB_RELAXATION_ARGS=--full` 使用原算例的初温、粒子数、步长、20 万/50 万步与第零格温度输出，生成两份原格式的理论对照曲线。

- `make test_coulomb_relaxation` compares the evolution of two electron temperatures and Spitzer electron–ion temperatures with theory. The short run uses 32 768 particles per group and 512 steps. It starts theory from the measured initial temperatures averaged across cells and checks both groups every 128 steps against a 2% relative-error limit. `make test_coulomb_relaxation COULOMB_RELAXATION_ARGS=--full` restores the original initial temperatures, particle counts, step sizes, runs of 200 000 and 500 000 steps, and cell-zero temperature output. It produces both curves in their original format.

论文完整算例的源码为 [paper_temperature_relaxation.cpp](paper_temperature_relaxation.cpp)，直接保留 `main` 的 `test_doubleT()` 和 `test_spitzer_relaxation()`。两者分别在 build 目录写出 `coulomb_doubleT_validation.plt` 与 `coulomb_spitzer_validation.plt`。保留原随机初始化、采样时机和列名，求解器使用当前库的混池实现。默认短程回归的源码为 [test_temperature_relaxation.cpp](test_temperature_relaxation.cpp)。

The full driver retains `test_doubleT()` and `test_spitzer_relaxation()` from `main`. They write `coulomb_doubleT_validation.plt` and `coulomb_spitzer_validation.plt`, respectively, in the build directory. The driver preserves the original random initialization, sampling times and column names and uses the current pooled solver. [Full paper driver](paper_temperature_relaxation.cpp) · [Default short regression](test_temperature_relaxation.cpp)
