[English](coding-style_en.md) | 中文

# PSuM 代码风格规范

## 概述

本文档描述了 PSuM 项目核心部分的代码风格。建议开发中维持此风格。


## 1. 文件结构与命名空间

### 1.1 文件路径与命名空间映射

**规则**：文件路径（src 目录下）与命名空间结构对应，命名空间和文件名均使用蛇形命名法（小写字母和下划线）。

**示例**：
```
src/tag/foundation.hpp                      -> namespace psum { namespace tag { namespace foundation { ... } } }
src/field/device_field.hpp                  -> namespace psum { namespace field { ... } }
src/particle_container/device_vector.hpp    -> namespace psum { namespace particle_container { ... } }
src/serialization/object_manager.hpp        -> namespace psum { namespace serialization { ... } }
```

**说明**：
- 如果文件创建了一组类型、模板或函数，则这些内容组织为命名空间，命名空间名称与文件名相同。
- 如果文件主要用于创建单个类型或函数，则文件名与类型名相同，没有命名空间。
- 简而言之：文件夹总对应命名空间；文件对应命名空间，或重要的类型/函数。

### 1.2 头文件保护

**规则**：全大写，用下划线连接路径和文件名

```
src/tag/foundation.hpp                      -> #ifndef PSUM_TAG_FOUNDATION_HPP
src/field/device_field.hpp                  -> #ifndef PSUM_FIELD_DEVICE_FIELD_HPP
src/particle_container/device_vector.hpp    -> #ifndef PSUM_PARTICLE_CONTAINER_DEVICE_VECTOR_HPP
```

### 1.3 头文件内容布局

**规则**：头文件内容按顺序排列，依次为：
```cpp
#ifndef HEADER_GUARD
#define HEADER_GUARD

// introduce library dependencies first
#include <library>
// then project headers
...
#include "file/in/project.hpp"
...

// remeber write content in 'psum'
namespace psum {

// and right module namespace
namespace _module_you_are_writing {

    // namespace/class/template declarations
    // whatever you want to declare in this module
    ...

}

}

#endif
```



## 2. 命名规范

### 2.1 基本原则

**规则**：大部分时候使用蛇形命名法（小写字母和下划线）。

**示例**：
- 文件名：`foundation.hpp`, `device_field.hpp`, `particle_group.hpp`
- 类/结构体：`device_field`, `particle_container`, `tagged_struct`
- 函数：`get_access()`, `to_host()`, `for_each()`
- 命名空间：`psum`, `tag`, `field`, `serialization`

### 2.2 私有成员命名约定

**规则**：成员变量和私有方法使用下划线后缀，表示不应从外部访问。

**说明**：这是一种"惩罚性命名"机制，下划线后缀作为一种视觉提示，提醒使用者这些成员不应从外部直接访问。

```cpp
class device_vector {
private:
    mutable sycl::queue q_;      // 成员变量带下划线后缀
    T* data_;
    size_t* size_;
    bool* overflow_;
    size_t capacity_;

    void set_size_(size_t new_size) const;  // 私有方法带下划线后缀
    size_t get_size_() const;

public:
    // 公共接口不使用下划线后缀
    size_t size() const;
    T* data() const;
};
```

**例外**：类内部的类型别名（using / typedef）**不遵守**此规则，因为那些类型别名通常被外部使用。即使私有类型别名也不使用下划线后缀。

```cpp
class device_vector {
public:
    using value_type = T;           // 类型别名不使用下划线后缀
    using acc_type = device_vector_acc<T>;

private:
    T* data_;                      // 成员变量使用下划线后缀
};
```

### 2.3 枚举值

**规则**：枚举值使用小驼峰命名法（camelCase）。

**示例**：
```cpp
enum class var_loc {
    cellCentered,   // 小驼峰
    faceCentered,
    edgeCentered,
    nodeCentered
};
```

### 2.4 模板参数

**规则**：模板参数通常使用首字母大写的名称。

**说明**：
- 使用描述性名称：`T`, `Func`, `Data`, `Item`, `Tag`, `Type`, `Scalar`, `Dimension`。
- 对于那些有物理意义的模板参数，不建议使用`T`这类单字母名称。
- 如果模板参数与类型内部的类型别名相同，使用前置下划线。

**示例**：模板参数名与类成员名不会产生冲突时，不使用下划线前缀。

```cpp
// 大多数模块
template <typename T>
class device_vector { ... };

template <typename Func, typename Data>
concept handler_to_device_func = ...;

template <typename Item, typename First, typename... Rest>
struct index_in_tuple { ... };

template <typename Type>
concept has_tag_name = ...;
```

**示例**：如果模板参数与类型内部的类型别名相同，使用前置下划线。这见于模板参数精确对应内部的类型别名的情况。

```cpp
// field 模块示例
template <int _Dimension, var_loc _location, typename _Scalar, int _QuantitySize = 1>
class device_field {
public:
    // 模板参数：_Dimension, _location, _Scalar, _QuantitySize (带下划线前缀)
    // 因为类内部有对应的类型别名，所以模板参数使用下划线前缀

    static constexpr int Dimension = _Dimension;      // 类内类型别名（大写开头）
    static constexpr var_loc Location = _location;
    static constexpr int QuantitySize = _QuantitySize;

    using Scalar = _Scalar;                            // 类内类型别名（大写开头）
    using Value = ...;
    using Position = ...;
    using Grid = simple_grid<Dimension>;
};
```

**示例**：如果模板参数与类型内部的类型别名不同，则互不影响。

```cpp
// particle_container 模块 - 不使用下划线前缀
template <typename T>
class device_vector {
    using value_type = T;              // 直接使用，不使用下划线前缀
    using acc_type = device_vector_acc<T>;
};
```

### 2.5 宏定义

**规则**：宏定义使用全大写，单词间用下划线连接。

```cpp
#define PSUM_UNIQUE_PTR_TYPE(x) std::unique_ptr<x[]>
#define PSUM_MATCH_TYPE_MEMSET(x) {std::string(#x), [](size_t size){return vPtr( std::make_unique<x[]>(size) );}}
#define DEFAULT_SEED 1
```

## 3. 代码组织

### 3.1 模块

**规则**：`src`目录下的每个子目录对应一个模块。

**示例**：
```
src/tag/                    -> 标签模块
src/field/                  -> 场量模块
src/particle_container/     -> 粒子数据结构模块
src/serialization/          -> 序列化模块
```

**说明**：
- 同一目录下的文件属于同一模块，共享相同的命名空间并形成完整的功能。
- 模块之间尽量不要有依赖关系；有依赖关系的情况，对应的文件放在依赖方而不是被依赖方的目录下。

### 3.2 模块入口文件

**规则**：每个模块都有一个入口文件，以简化外部引入。

**示例**：
```cpp
src/tag.hpp                    // tag模块入口
src/particle_container.hpp     // particle_container模块入口
src/field.hpp                  // field模块入口
src/serialization.hpp         // serialization模块入口
```

## 附录：命名速查表

| 类型 | 命名规则 | 示例 |
|------|---------|------|
| 文件名 | 小写+下划线 | `foundation.hpp`, `device_field.hpp` |
| 类/结构体 | 小写+下划线 | `device_field`, `tagged_struct` |
| 公共函数 | 小写+下划线 | `get_access()`, `for_each()` |
| **成员变量** | **小写+下划线 + 下划线后缀** | **`data_`**, **`size_`**, **`capacity_`** |
| **私有方法** | **小写+下划线 + 下划线后缀** | **`set_size_()`**, **`get_size_()`** |
| 命名空间 | 小写 | `psum`, `tag`, `field` |
| 枚举值 | 小驼峰（camelCase） | `cellCentered`, `faceCentered` |
| 类型别名 | 小写 | `value_type`, `acc_type` |
| 模板参数（默认） | 首字母大写 | `T`, `Func`, `Type` |
| **模板参数（与类型别名冲突）** | **下划线前缀** | **`_Dimension`**, **`_Scalar`** |
| 宏定义 | 全大写+下划线 | `PSUM_UNIQUE_PTR_TYPE`, `DEFAULT_SEED` |
