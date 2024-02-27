# Executors

Executor (`IExecutor`) is an execution engine of a subgraph that can execute inference for the subgraph. It is the result of a `Subgraph` compilation. Compared to common programming language tools, it is like an interpreter with code to execute.

## Understanding models

We can think of an NNPackage model as a set of tasks with dependencies. In other words, it is a form of [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (more precisely, it is a set of DAGs, as we need multiple subgraphs to support control flow operations). And that is exactly the same concept with [Dataflow programming](https://en.wikipedia.org/wiki/Dataflow_programming).

That is, there are some input tensors that must be ready to run a operation. And the execution must be done in topological order. Here's the workflow for execution.

1. User gives model input tensors
2. Start execution
3. Perform the operations that are ready
4. Mark the tensors as ready that are made from the operations that was just performed
5. Check if there are some operations ready
    1. If yes, Go to 3
    2. Otherwise, Finish execution
6. User consumes data of model output tensors

We have 3 different types of executors in our codebase and they all are based on the above explanation. However, only `LinearExecutor` is official and the other two are experimental.

## Linear Executor

`LinearExecutor` is the main executor. As we know the model to run and the model graph does not change at runtime, we do not need to do the above steps 3-5 at runtime. During the compilation for Linear Executor, it sorts operations in topological order so we can just execute in that fixed order which means that it cannot perform the operations in parallel.

If the tensors are static, it also can analyze the lifetimes of the tensors and pre-allocate tensor memory with reusing memory between the tensors whose lifetimes do not overlap.

## Dataflow Executor (experimental)

Unlike `LinearExecutor`, `DataflowExecutor` does steps 3-5 at runtime. By doing it we can know which operations are available at a specific point. However this executor still executes the operations one at a time. Just choose any operation that is ready then execute, wait for it to finish then repeat. So there may be no advantage compared to `LinearExecutor` but `DataflowExecutor` is the parent class of `ParallelExecutor`. And `DataflowExecutor` can be used for profiling executions for the heterogeneous scheduler.

## Parallel Executor (experimental)

Just like `DataflowExecutor`, `ParallelExecutor` does steps 3-5 at runtime. One big difference is that it creates a `ThreadPool` for each backend for parallel execution (`ThreadPool` is supposed to have multiple threads, however for now, it can have only one thread). Multiple operations ready to execute can be executed in different backends at the same time, which could lead to some performance gain.
