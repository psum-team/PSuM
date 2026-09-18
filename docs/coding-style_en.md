English | [中文](coding-style.md)

# PSuM Coding Style Guide

## Overview

This document describes the code style for the core parts of the PSuM project. It is recommended to maintain this style during development.


## 1. File Structure and Namespaces

### 1.1 File Path and Namespace Mapping

**Rule**: File paths (under the `src` directory) correspond to the namespace structure. Both namespaces and file names use snake_case (lowercase letters and underscores).

**Examples**:
```
src/tag/foundation.hpp                      -> namespace psum { namespace tag { namespace foundation { ... } } }
src/field/device_field.hpp                  -> namespace psum { namespace field { ... } }
src/particle_container/device_vector.hpp    -> namespace psum { namespace particle_container { ... } }
src/serialization/object_manager.hpp        -> namespace psum { namespace serialization { ... } }
```

**Notes**:
- If a file creates a group of types, templates, or functions, they are organized into a namespace whose name matches the file name.
- If a file is primarily for creating a single type or function, the file name matches the type/function name, with no additional namespace.
- In short: folders always correspond to namespaces; files correspond to either a namespace or an important type/function.

### 1.2 Header Guards

**Rule**: All uppercase, with underscores separating path components and file name.

```
src/tag/foundation.hpp                      -> #ifndef PSUM_TAG_FOUNDATION_HPP
src/field/device_field.hpp                  -> #ifndef PSUM_FIELD_DEVICE_FIELD_HPP
src/particle_container/device_vector.hpp    -> #ifndef PSUM_PARTICLE_CONTAINER_DEVICE_VECTOR_HPP
```

### 1.3 Header File Content Layout

**Rule**: Header file content is arranged in the following order:
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



## 2. Naming Conventions

### 2.1 General Principle

**Rule**: Use snake_case (lowercase letters and underscores) in most cases.

**Examples**:
- File names: `foundation.hpp`, `device_field.hpp`, `particle_group.hpp`
- Classes/structs: `device_field`, `particle_container`, `tagged_struct`
- Functions: `get_access()`, `to_host()`, `for_each()`
- Namespaces: `psum`, `tag`, `field`, `serialization`

### 2.2 Private Member Naming

**Rule**: Member variables and private methods use a trailing underscore suffix, indicating they should not be accessed externally.

**Note**: This is a "punitive naming" mechanism. The trailing underscore serves as a visual reminder that these members should not be accessed directly from outside.

```cpp
class device_vector {
private:
    mutable sycl::queue q_;      // member variables with trailing underscore
    T* data_;
    size_t* size_;
    bool* overflow_;
    size_t capacity_;

    void set_size_(size_t new_size) const;  // private methods with trailing underscore
    size_t get_size_() const;

public:
    // public interfaces do not use trailing underscore
    size_t size() const;
    T* data() const;
};
```

**Exception**: Type aliases (using / typedef) inside a class **do not** follow this rule, since they are typically used externally. Even private type aliases do not use the trailing underscore.

```cpp
class device_vector {
public:
    using value_type = T;           // type alias without trailing underscore
    using acc_type = device_vector_acc<T>;

private:
    T* data_;                      // member variable with trailing underscore
};
```

### 2.3 Enum Values

**Rule**: Enum values use camelCase (lower camelCase).

**Examples**:
```cpp
enum class var_loc {
    cellCentered,   // camelCase
    faceCentered,
    edgeCentered,
    nodeCentered
};
```

### 2.4 Template Parameters

**Rule**: Template parameters typically use names with a capital first letter.

**Notes**:
- Use descriptive names: `T`, `Func`, `Data`, `Item`, `Tag`, `Type`, `Scalar`, `Dimension`.
- For template parameters with physical meaning, single-letter names like `T` are not recommended.
- If a template parameter name conflicts with an internal type alias, use a leading underscore.

**Example**: When template parameter names do not conflict with class member names, no leading underscore is used.

```cpp
// most modules
template <typename T>
class device_vector { ... };

template <typename Func, typename Data>
concept handler_to_device_func = ...;

template <typename Item, typename First, typename... Rest>
struct index_in_tuple { ... };

template <typename Type>
concept has_tag_name = ...;
```

**Example**: If a template parameter conflicts with an internal type alias, use a leading underscore. This occurs when the template parameter corresponds exactly to an internal type alias.

```cpp
// field module example
template <int _Dimension, var_loc _location, typename _Scalar, int _QuantitySize = 1>
class device_field {
public:
    // template parameters: _Dimension, _location, _Scalar, _QuantitySize (with leading underscore)
    // because the class has corresponding type aliases, the template parameters use a leading underscore

    static constexpr int Dimension = _Dimension;      // internal type alias (capitalized)
    static constexpr var_loc Location = _location;
    static constexpr int QuantitySize = _QuantitySize;

    using Scalar = _Scalar;                            // internal type alias (capitalized)
    using Value = ...;
    using Position = ...;
    using Grid = simple_grid<Dimension>;
};
```

**Example**: If the template parameter does not conflict with an internal type alias, they do not affect each other.

```cpp
// particle_container module - no leading underscore
template <typename T>
class device_vector {
    using value_type = T;              // used directly, no leading underscore
    using acc_type = device_vector_acc<T>;
};
```

### 2.5 Macro Definitions

**Rule**: Macro definitions use all uppercase letters with underscores separating words.

```cpp
#define PSUM_UNIQUE_PTR_TYPE(x) std::unique_ptr<x[]>
#define PSUM_MATCH_TYPE_MEMSET(x) {std::string(#x), [](size_t size){return vPtr( std::make_unique<x[]>(size) );}}
#define DEFAULT_SEED 1
```

## 3. Code Organization

### 3.1 Modules

**Rule**: Each subdirectory under `src` corresponds to a module.

**Examples**:
```
src/tag/                    -> tag module
src/field/                  -> field module
src/particle_container/     -> particle data structure module
src/serialization/          -> serialization module
```

**Notes**:
- Files in the same directory belong to the same module, share the same namespace, and form a complete unit of functionality.
- Modules should generally not have dependencies on each other. When dependencies do exist, the corresponding files should be placed in the dependent party's directory, not the dependency's.

### 3.2 Module Entry Files

**Rule**: Each module has an entry file to simplify external inclusion.

**Examples**:
```cpp
src/tag.hpp                    // tag module entry
src/particle_container.hpp     // particle_container module entry
src/field.hpp                  // field module entry
src/serialization.hpp         // serialization module entry
```

## Appendix: Naming Quick Reference

| Category | Naming Rule | Examples |
|----------|-------------|----------|
| File names | lowercase + underscores | `foundation.hpp`, `device_field.hpp` |
| Classes/structs | lowercase + underscores | `device_field`, `tagged_struct` |
| Public functions | lowercase + underscores | `get_access()`, `for_each()` |
| **Member variables** | **lowercase + underscores + trailing underscore** | **`data_`**, **`size_`**, **`capacity_`** |
| **Private methods** | **lowercase + underscores + trailing underscore** | **`set_size_()`**, **`get_size_()`** |
| Namespaces | lowercase | `psum`, `tag`, `field` |
| Enum values | camelCase | `cellCentered`, `faceCentered` |
| Type aliases | lowercase | `value_type`, `acc_type` |
| Template parameters (default) | Capitalized first letter | `T`, `Func`, `Type` |
| **Template parameters (conflict with type alias)** | **leading underscore** | **`_Dimension`**, **`_Scalar`** |
| Macro definitions | ALL_CAPS + underscores | `PSUM_UNIQUE_PTR_TYPE`, `DEFAULT_SEED` |
