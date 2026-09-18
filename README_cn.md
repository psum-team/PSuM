language: [English](README.md) | 中文

# PSuM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PSuM 是一个面向 PIC（Particle-in-Cell）算法的 C++/SYCL 框架。它希望用同一份程序表达粒子、场、边界和碰撞等计算逻辑，并尽量适配 CPU、GPU 等不同硬件。

目前，PSuM 以源码形态提供框架库功能：功能以源代码形式给出，随本仓库的 `include/` 与 `src/` 需要一起编译进你的程序——PSuM 不以预编译二进制或可安装 C++ 包的形式分发。

官方网站：<https://psum.space>


## 安装配置

如果你的环境可控（例如个人电脑或工作站），优先使用 Docker：

- [Docker 安装](docker_install/readme.md)

如果 Docker 不适用，再阅读手动配置文档：

- [依赖项安装](docs/dependency.md)
- [环境配置与构建](docs/building.md)

配置完成后，可以先运行测试脚本确认环境是否可用：

```bash
./test/runCheck.sh
```

默认是冒烟运行：所有目标都会编译并启动，超过（较短的）默认时限仍在运行的测试会按超时如实报告，而不计为失败。如需严格的完整运行，请使用：

```bash
TIMEOUT=20 RUNCHECK_STRICT=1 bash test/runCheck.sh
```

在 CI 中，`test/rcheck_skip_targets.list` 中列出的长耗时目标会被跳过，其余目标应在时限内完成。

## 首个示例

[示例总览](docs/examples.md) 介绍了仓库中内置的示例。
Tour 系列更关注"能跑、能看、能理解"，而不是完整的物理模型。
Benchmark 系列更适合对 PIC 方法有比较充分理解的开发者。

## 目标

PIC（Particle-in-Cell）方法天然包含大量粒子级并行计算，适合在 GPU 等并行硬件上执行——合理实现后，一个高端消费级 GPU 在某些 PIC 任务中的吞吐能力可以接近上百个 CPU 核心。但是，直接面向 GPU 编写 PIC 程序也会带来额外的工程负担：需要掌握 SYCL/CUDA 等并行编程模型，GPU 程序的调试和验证更复杂，CPU 上自然的内存访问模式在 GPU 上未必高效，跨硬件厂商的迁移成本也较高。PSuM 的目标是在保留高性能潜力的同时，尽量降低这些工程复杂度。它基于 SYCL 编程模型，使同一份 C++ 代码可以适配 CPU、GPU 等不同平台。

- **屏蔽硬件差异** —— 目标不只是"让 PIC 在 GPU 上运行"，而是让 PIC 程序的表达尽量独立于具体硬件。用户用一份代码描述粒子、场、边界和碰撞逻辑，由框架负责屏蔽部分底层硬件差异。
- **降低高性能实现的复杂度** —— PIC 程序的性能瓶颈并不总在粒子推进或插值环节，也可能出现在 Poisson 方程、隐式 Maxwell 方程或其他线性系统求解上。PSuM 通过可插拔的求解器后端，把物理模型和具体的线性代数实现分离，让用户可以根据硬件和问题规模选择合适的后端。框架内还集成了高性能的相交检测、碰撞对生成等算法。
- **分层暴露复杂度** —— PSuM 不假设所有用户都需要理解框架内部实现。文档和接口大致分为三层：**应用层**直接运行示例或基准算例，快速获得可检查的结果；**算法层**使用粒子容器、场量、边界、求解器和碰撞模块编写自定义算例；**核心层**替换或扩展粒子边界处理、碰撞模型、求解器后端等内部机制。

## 文档地图

- 想快速上手：阅读 [示例总览](docs/examples.md)，跟随 tour 系列逐步修改和运行代码。
- 想写自己的程序：阅读 [编程指南](docs/programming.md)，了解 tag、粒子容器、场量、求解器和边界系统。
- 想理解 PIC 背景：阅读 [PIC 简介](docs/pic.md)。
- 想参与框架开发：阅读 [框架设计](docs/design.md) 和 `docs/detailed_design/` 下的设计文档。
- 辅助脚本 `repo_heatmap.py` 可生成 `src/` 的代码分布热力图（依赖：pandas、plotly、matplotlib；PNG 导出另需 Kaleido）。
更完整的导航见 [PSuM 文档索引](docs/index.md)。

## 贡献

PSuM 目前以核心团队为主导进行开发。最有帮助的参与方式是提交 Issue——报告 Bug、提出功能建议、讨论设计思路都非常欢迎。我们暂不直接接受未经沟通的 Pull Request，详见[贡献指南](CONTRIBUTING_cn.md)。

## 引用 PSuM

如果你在研究中使用了 PSuM，请按 [CITATION.cff](CITATION.cff) 中的元数据进行引用。

## 许可证

PSuM 以 [MIT 协议](LICENSE)开源。`third_party/` 下的第三方组件保留其各自原始许可证，详见[第三方声明](THIRD_PARTY_NOTICES.md)。
