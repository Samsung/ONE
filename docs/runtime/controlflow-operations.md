# Controlflow Operations

We call the `If` and `While` operations "Controlflow operations". These operations are special. Instead of computing data, they are used to invoke another subgraph and return back which constitutes conditional/iterations work in dataflow models.

## Defining controlflow operations

As we use Tensorflow Lite schema (or Circle which is based on TF Lite), the runtime follows the way TF Lite does. The details are stated in the [Control Flow in TensorFlow Lite](https://github.com/tensorflow/community/blob/master/rfcs/20190315-tflite-control-flow.md) RFC document.

Controlflow operations from NN API are not yet supported. But we expect that they can be enabled in a similar way.

## Implementation

### Graph representation

`onert` internally has its representation for controlflow operations and subgraphs. It is straightforward as it is pretty much isomorphic with the schema. The `onert`'s in-memory model contains multiple subgraphs and the controlflow operations have same parameters (subgraph indices), just like TF Lite schema has.

### Execution

The `controlflow` backend is a built-in backend to support these controlflow operations. This backend is special as it has access to `onert` core's executor manager (`ExecutorMap`) so it can invoke/return a subgraph. This backend has implementations for `If` and `While` operations and they make use of the access to executor manager.

An `Executor` has two different ways to execute depending on if it is the initial execution or invoking a subgraph from a controlflow operation.

- Executing the primary subgraph
    - Pass user-given tensors as the subgraph inputs and outputs
- Executing a subgraph for controlflow operations
    - Pass controlflow operation inputs tensors as the subgraph inputs
    - Pass the subgraph outputs as controlflow operation outputs

#### Kernel Implementation

Here is a brief explanation what the kernels do, which is quoted from [Control Flow in TensorFlow Lite](https://github.com/tensorflow/community/blob/master/rfcs/20190315-tflite-control-flow.md).

> * `If` : Check the condition input and invoke one of the 2 subgraphs.
> * `While` :
>     * Invoke the condition subgraph. Break out the loop if the result is false.
>     * Invoke the body subgraph, use the output as the input of the next iteration.

Invoking a subgraph needs to pass the operation's inputs to the subgraph inputs. And Returning back needs to pass the subgraph outputs to the operation outputs.

When invoking a subgraph and returning back, the current kernel implementation makes a copy of all the subgraph inputs and outputs. This is going to be optimized to minimize redundant copies.
