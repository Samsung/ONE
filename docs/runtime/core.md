# Core

Runtime Core is a compilation/execution engine for neural network models.

## Modules

Runtime Core has four modules. These are namespaces as well as directory names in `/runtime/onert/core/src/`.

- `ir`  stands for Intermediate Representation which contains Neural Network Graph data structures
- `compiler` converts IR to executable format
- `exec` is execution module which is the result of compilation
- `backend` is backend interface

### Module `ir`

This module contains data structures of pure Neural Network models. The models from NN Packages or NN API are converted to these structures.

- `Subgraphs` is the entire neural network model which is a group of subgraphs
- `Subgraph` consists of operands and operations
- `Operand` (a.k.a. Tensor) has shape, data type, data and references to operations
- `Operation` (a.k.a. Operator) has operation type, params and references to operands

`Operand` and `Operation` are graph nodes. References to operations and operands are graph edges.

`Subgraphs` represents the whole model. It could have more than one `Subgraph` to support control flow operations. Those operations can make calls to another subgraph and when the execution on another subgraph is done it gets back to previous subgraph execution with returned operands.

All `Graph`s are a [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph) so once model inputs are given we can run it in topological order.

Here's a figure of how those data structures are oraganized.

![Core](core-figure-ir.png)
### Module `compiler`

`Compiler` is the main class of this module. Everything starts from it.

What it does is making models executable. It schedules run order and assigns a backend for each operation. Here are major tasks of compilation.

#### 1. Lowering

In Lowering, the compiler assigns a [backend](#) for each operation. A backend is assigned to an operation means that the operation will be run with the assigned backend's kernel.

There is a scheduler that allows the user manually specify backends via compiler options. There is another scheduler that automatically assigns backends based on profiling info.

#### 2. Tensor Registration

Each backend manages its tensors. In this phase operand informations get registered to the corresponding backend. This will be used generating tensor objects.

##### Q. What are differences between "operand" and "tensor"?

In ONE Runtime, "operand" refers to an operand in a NN model. While "tensor" includes all "operand" info plus actual execution info like actual buffer pointer. In short, "operand" is for `ir`, "tensor" is for `backend`.

#### 3. Linearization (Linear Executor Only)

For Linear Executor, it needs to be linearized before execution. Linearizaton means sorting operations in a topological order. It saves execution time since resolving next available operations after every operation is not needed at execution time. Also it makes plans for tensor memory. It can save some memory space by reusing other operands' space that does not overlap lifetime. Also all allocations are done at compile time (after [4. Kernel Generation](#4.-kernel-generation)) which saves execution time.

#### 4. Kernel Generation

A backend is assigned for each operation. In this phase kernels for each operation are generated.

#### 5. Create Executor

With generated tensors and kernels, the compiler creates executor objects. There are 3 types of executors are supported - Linear, Dataflow and Parallel. Linear executor is the default executor and Dataflow Executor and Parallel Executor are experimental.

For more about executors, please refer to [Executors](#) document.

### Module `exec`

As a result of compilation, `Execution` is created. Users can set input and output buffers then finally run it!

### Module `backend`

Backends are plugins and they are loaded dynamically(via `dlopen`). So this module is a set of interface classes for backend implementation. `compiler` can compile with variety of backends without knowing specific backend implementation.

For more, please refer to [Backend API](#) document.
