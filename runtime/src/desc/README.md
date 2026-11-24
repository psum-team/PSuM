# 对象描述符 / Object descriptors

对象描述符用来对C++中的一组相关类型——通常由模板创建出来——的实例建立 C 接口。
例如，std::array可以从长度和储存对象的类型派生出许多实际类型: `array<int, 1>`, `array<int, 2>`, ..., `array<double, 1>`, ...。
c++中的模板用于编译期生成代码，c接口无法完全传达这一点。然而，现实中一个模板所产生的实例是有限的。因此，我们可以对实例化时的模板参数进行编码并在运行时指代这些不同类型。
在下面的记号中，我们用 X_desc 表示模板X派生的一组类型的对象的描述符。
相关的约定如下。

An object descriptor gives the C interface to the instances of a group of related C++ types, usually produced from a template.
For example, `std::array` derives many concrete types from the length and the type of the stored elements: `array<int, 1>`, `array<int, 2>`, ..., `array<double, 1>`, ....
A C++ template generates code at compile time, which a C interface cannot fully convey. In practice, however, the instances produced by one template are finite in number. We can therefore encode the template arguments of an instantiation and refer to these distinct types at run time.
In the notation below, `X_desc` denotes the descriptor of the objects of the group of types derived from the template X.
The conventions are as follows.

## 原则 / Principles

1、描述符由 `name`、TYPE 不变量、VALUE 不变量组成。`name` 是对象的身份；TYPE 不变量需完全决定对象的 C++ 类型，通常可对应上模板参数；VALUE 不变量是对象的只读设置。
描述符需要可移植、二进制兼容。描述符是纯 C 结构体；定长数据用值数组；仅`name`字段可用裸指针。

Principle 1, three parts. A descriptor consists of `name`, TYPE invariants and VALUE invariants. `name` is the object's identity; the TYPE invariants must determine the C++ type of the object completely and usually correspond to the template arguments; the VALUE invariants are read-only settings of the object.
A descriptor must be portable and binary compatible. A descriptor is a plain C struct; fixed-size data is held in value arrays; only the `name` field may be a bare pointer.

2、对于 X_desc ，需提供运行期功能六项：
2.1 按 `{X, name}` 判断对象是否存在；
2.2 保证 `{X, name}` 至多一个对象；
2.3 判断 X_desc 的实例 `d` 是否真实地指向一个对象（即name, TYPE和VALUE不变量均对应）；
2.4 如 `{X, name}` 指向存在的对象，给出该对象对应的真实的对象描述符 `d` (从而，获得 TYPE 和 VALUE 不变量)；
2.5 如果 `d` 的 `name` 尚未使用，尝试将 `d` 描述的对象构造出来，使得 `d` 变成真实的；
2.6 按 `name` 和其他参数创建对象，返回一个真实的 `d`。

Principle 2, six run-time functions must be provided for X_desc:
2.1 tell whether an object exists for `{X, name}`;
2.2 guarantee that there is at most one object for `{X, name}`;
2.3 tell whether an instance `d` of X_desc is authentic, that is, whether its name, TYPE invariants and VALUE invariants all match an object;
2.4 if `{X, name}` refers to an existing object, give the authentic descriptor `d` of that object (and with it its TYPE and VALUE invariants);
2.5 if the `name` of `d` is not yet used, try to construct the object that `d` describes, so that `d` becomes authentic;
2.6 create an object from `name` and further arguments, and return an authentic `d`.

3、create 过程的返回值约定。如名字下没有对象：创建并返回非异常结果。同名同类但 TYPE 不变量不同：返回类型不变量冲突对应的错误码。TYPE 相同但 VALUE 不变量不同：返回值不变量冲突对应的错误码。完全相同的对象：返回对象已存在的错误码。无论结果如何，`d` 都被回填为真实不变量。

Principle 3, return values of create. No object under the name: the object is created and non-abnormal result is returned. Same name and kind but different TYPE invariants: the error code for a TYPE-invariant conflict is returned. TYPE invariants equal but VALUE invariants differ: the error code for a VALUE-invariant conflict is returned. Identical object: the error code for an object that already exists is returned. Whatever the outcome, `d` is overwritten with the real invariants.
