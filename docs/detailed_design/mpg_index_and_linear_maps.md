[English](mpg_index_and_linear_maps_en.md) | 中文

## MPG 索引与索引映射约定

这里用MP作为multi-patch的缩写。MPG = Multi-Patch Grid。
在MP-field中, 场量数据用存在重复的原始index顺序储存，且MPG-solver也接收采用原始index编号的指针数据。
然而，在MPG-solver内部处理中, 为了使用FEM求解器，会先将其转化为无重复的dof顺序。
相关概念如下。

### MPG 原始 index

`index` 是 MPG 中网格元素的原始线性编号, 其计算方式是简单的:

```text
global_index = patch_node_offsets[p] + local_node_index
```

其中patch_node_offsets[p]表示了前p个patch中的网格元素总数，local_node_index表示了网格元素在patch中的编号。
local_node_index 的计算方法即是simple_grid中的c2i/n2i函数的计算方法。
这种索引也是MP-field中原始数组中的储存顺序。

对于cell index而言, 每个网格元素对应着唯一的index, 但对于node index, 相邻的patch中，不同index可能对应着相同的网格元素。

### DOF Index

对于前面描述的情况，可以定义一个去重的编号: 当一个原始index指向不同于更早index的网格元素时，该index被分配到一个新的dof编号。
(dof is short for degree of freedom)
如果一个index对应于更小的index所指向的网格元素，则具有与之相同的dof编号。

`duplicate_map<Dim>` 负责描述 MPG 原始 node index 和 dof index 之间的关系, 它可以仅由mpg对象生成。
`duplicate_map`包含:

```cpp
size_t n_index;
size_t n_dof;
std::vector<size_t> index_to_dof;
Eigen::SparseMatrix<double> P;
//...
```

其中 `index_to_dof[i]` 表示第 `i` 个 MPG node index 对应的 dof。
dof编号总小于index编号，且index编号唯一地对应着dof编号（反之则不成立）。

### 三种线性映射

#### Scatter

该映射将dof索引映射到原始index索引。该映射由 `duplicate_map` 中的 `P` 矩阵表示。
`P`满足:

```text
P(i, index_to_dof[i]) = 1
```

这是一个行比列多的矩阵，因为每个dof对应多个index。

典型用法：
```text
f = P * f_dof
```

含义：把去重后的 dof 数据展开回 MPG index 数据。共享同一 dof 的多个 MPG index 会得到相同数值。
典型用途：FEM solver 得到 `phi_dof` 后，写回 MPG 视角的 `phi_index`。


#### Merge

该映射将多个原始index索引映射到一个dof索引并进行累加，并将结果广播到所有原始index。
该映射可由 `duplicate_map` 中的 `P` 矩阵及其转置计算。
典型用法：

```text
f_merged = (P * P.transpose()) * f
```

该映射由函数`build_merge_matrix(const duplicate_map<Dim>& dm)`直接给出。
该映射常用于在MP-field中表示网格元素相关的广延量，如体积等。
该映射不是幂等的，连续使用会导致关联的index上数据反复累加。

#### Gather

该映射将对应于将一个dof的最大的原始index索引映射到dof索引。该映射是`P`矩阵转置后, 每行保留最后一个非零元素留下的矩阵。

```text
index -> dof (若index是对应于这dof的最大的)
```

矩阵形式：

```text
x_dof = G * x_index
```

如果多个 index 对应同一个 dof，dof会被映射为最后一个关联的 index 处的数据。
它的作用等价于:
```cpp
for (size_t i = 0; i < dm.n_index; i++)
    dof_data[dm.index_to_dof[i]] = index_data[i];
```

### MPG 求解器中的映射过程

在 MPG solver 内部设计gather/scatter映射，并统一视为 SpMV：

- `gather_spmv_calculator_` 执行 `G * x_index`
- `scatter_spmv_calculator_` 执行 `P * x_dof`

用户通过：
```cpp
set_queue(queue) // 在solve前调用
```
指定MPG求解器及其内部FEM求解器使用的SYCL queue。MPG不暴露单独的SpMV类型设置接口，以保证gather/scatter的device buffer与SpMV calculator使用同一queue。

在每次solve时, 传入和传出均使用原始index顺序的数据（与MP-field中保持一致, 以便直接传入裸指针求解），因此:
```text
source_dof = gather_spmv_calculator_ * source_index
fem_solver.solve(solution_dof, source_dof)
solution_index = scatter_spmv_calculator_ * solution_dof
```
执行两次 SpMV 计算.

MPG solver不会用到merge映射。merge映射可以合并关联 index 分别统计到的电荷量, 再除以体积得到电荷密度，形成求解器的输入。
