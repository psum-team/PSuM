[English](index_en.md) | 中文

## PSuM 文档索引

PSuM 的文档按阅读深度分为三层。第一次使用时，建议先跑示例，再回头阅读概念和设计文档。

### 1. 快速跑起来

- [Docker 安装](../docker_install/readme.md)：推荐的环境准备方式，适合个人电脑、工作站或其他可控环境。
- [依赖项安装](dependency.md) 与 [环境配置与构建](building.md)：Docker 不适用时再阅读，内容更偏环境和工具链细节。
- [示例总览](examples.md)：介绍 `example/` 下的 tour 互动算例，以及 `application/benchmark/` 下的基准算例。

### 2. 学会使用 PSuM

- [PIC 简介](pic.md)：Particle-in-Cell 方法的基本背景。
- [PSuM 中的 C++](cpp-in-psum.md)：补充 C++ 模板、lambda 和 SYCL 写法。
- [编程指南](programming.md)：系统整理 PSuM 的基础 API，包括 tag、粒子容器、场量、边界、求解器等。

### 3. 理解背景和参与开发

- [框架设计](design.md)：面向贡献者和高级用户的设计说明。
- `docs/detailed_design/`：复杂模块的细化设计文档，例如求解器后端。
- [代码风格](coding-style.md)：参与开发前建议阅读。
