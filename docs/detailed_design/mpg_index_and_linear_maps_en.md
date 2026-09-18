English | [中文](mpg_index_and_linear_maps.md)

## MPG Indexing and Index-Mapping Conventions

Here MP stands for multi-patch. MPG = Multi-Patch Grid.
In an MP-field, field data is stored in the original index order, which contains
duplicates, and the MPG solver also receives pointer data using the original
index numbering. Internally, however, the MPG solver first converts this to a
duplicate-free DOF order so that the FEM solver can be used. The relevant
concepts are as follows.

### MPG original index

`index` is the original linear numbering of a grid element in the MPG; it is
computed simply as:

```text
global_index = patch_node_offsets[p] + local_node_index
```

where `patch_node_offsets[p]` is the total number of grid elements in the first
`p` patches and `local_node_index` is the element's number within its patch.
`local_node_index` is computed in the same way as the `c2i`/`n2i` functions in
`simple_grid`. This index is also the storage order of the original arrays in
the MP-field.

For cell indices, each grid element corresponds to a unique index, but for node
indices, different indices in adjacent patches may correspond to the same grid
element.

### DOF Index

For the situation above, a deduplicated numbering can be defined: when an
original index points to a grid element different from all earlier indices, it
is assigned a new DOF number (DOF is short for degree of freedom). If an index
corresponds to a grid element pointed to by a smaller index, it has the same DOF
number as that index.

`duplicate_map<Dim>` describes the relationship between MPG original node
indices and DOF indices; it can be generated from the `mpg` object alone.
`duplicate_map` contains:

```cpp
size_t n_index;
size_t n_dof;
std::vector<size_t> index_to_dof;
Eigen::SparseMatrix<double> P;
//...
```

where `index_to_dof[i]` is the DOF corresponding to the `i`-th MPG node index.
A DOF number is always smaller than its index numbers, and an index uniquely
corresponds to a DOF (the converse does not hold).

### Three linear maps

#### Scatter

This map maps DOF indices to original index indices. It is represented by the
`P` matrix in `duplicate_map`. `P` satisfies:

```text
P(i, index_to_dof[i]) = 1
```

It has more rows than columns, because each DOF corresponds to multiple indices.

Typical usage:

```text
f = P * f_dof
```

Meaning: expand the deduplicated DOF data back to MPG index data. Multiple MPG
indices sharing the same DOF get the same value. Typical use: after the FEM
solver obtains `phi_dof`, write it back to `phi_index` in the MPG view.

#### Merge

This map maps several original indices to one DOF index, accumulates them, and
broadcasts the result to all original indices. It can be computed from the `P`
matrix in `duplicate_map` and its transpose. Typical usage:

```text
f_merged = (P * P.transpose()) * f
```

This map is provided directly by the function
`build_merge_matrix(const duplicate_map<Dim>& dm)`. It is often used to represent
extensive quantities associated with grid elements (such as volume) in the
MP-field. The map is not idempotent; repeated application keeps accumulating data
on the associated indices.

#### Gather

This map maps, for each DOF, the largest original index corresponding to that
DOF to the DOF index. It is the matrix obtained by transposing `P` and keeping
the last nonzero entry in each row.

```text
index -> dof (if index is the largest corresponding to this dof)
```

Matrix form:

```text
x_dof = G * x_index
```

If several indices correspond to the same DOF, the DOF is mapped to the data at
the last associated index. It is equivalent to:

```cpp
for (size_t i = 0; i < dm.n_index; i++)
    dof_data[dm.index_to_dof[i]] = index_data[i];
```

### Map process in the MPG solver

Inside the MPG solver, gather/scatter maps are designed and uniformly treated as
SpMV:

- `gather_spmv_calculator_` performs `G * x_index`
- `scatter_spmv_calculator_` performs `P * x_dof`

The user specifies the SYCL queue used by the MPG solver and its internal FEM
solver via:

```cpp
set_queue(queue) // call before solve
```

The MPG does not expose a separate SpMV-type setting interface, so that the
gather/scatter device buffers use the same queue as the SpMV calculators.

On each solve, both input and output use original index order data (consistent
with the MP-field, so that raw pointers can be passed directly to the solver),
therefore:

```text
source_dof = gather_spmv_calculator_ * source_index
fem_solver.solve(solution_dof, source_dof)
solution_index = scatter_spmv_calculator_ * solution_dof
```

Two SpMV operations are performed.

The MPG solver does not use the merge map. The merge map can combine the charges
accumulated separately at associated indices, then divide by volume to obtain the
charge density that forms the solver input.
