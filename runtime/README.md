### runtime 子项目的目标

该子项目的目标是为PSuM框架提供：
- 细粒度的 c-api，从而形成PIC的 modeling language 及实现
- 基于上一点，形成动态链接库形态和可执行程序形态的 runtime
- 基于动态链接库，形成 python 绑定层，使PSuM支持 python 调度

### 基本设计

子项目需形成的主要代码对象是：
- psum_runtime_api.h: 一组可移植的c函数接口和结构体定义，用以表达 modeling language 和必要的I/O及资源生命周期的控制
- psum_engine.hpp: 一个c++类型，用于简化c-api的对应实现，亦可作应用框架之用
- psum_runtime.cpp: 通过对engine的浅封装实现c-api；so的编译单元
- psum_exec.cpp: 仅基于c-api和so形成可执行程序
- pypsum.cpp: 形成python绑定

其中, psum_runtime.cpp 依赖PSuM编译工具链; exec仅能依赖不超过c++11的标准编译工具链; pypsum.cpp依赖pybind11工具链。

需要支持基础场量上的PSuM的全部功能；batch-PIC, multi-patch field 等实验性特性暂时不进入路线图。

- functional features
[x] generic field/species manager
[x] standard PIC deposit and push
[x] basic poisson solver config
[] generic poisson solver config
[] limited and practical material-set and generic particle boundary geometry
[] limited and practical MCC module catalogue
[] limited DSMC module catalogue

- non-functional features
[] more generic object manager, including other common types
[] callback function interface
[] interface expansion convention
[] runtime test system