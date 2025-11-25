## Model IR (MIR)

### Purpose
This library exposes **NNC**'s model IR to the outer tools (currently `Mirunner`).

### Design philosophy

**MIR** was designed to support a multiple-frontend NN compiler/optimizer.

### Function

The high level overview of **MIR** is:
* operations are a composition of their `inputs`, `outputs` and 
special attributes specific to different operation types.
* operations can have multiple inputs and multiple outputs,
 each output can be an input to more than one operation 
 (can be used in more than one operation).
* the kernel tensors are represented by `ConstantOp` and
 are linked to operations via `Input` objects.

Mir has a protobuf serializer/deserializer for shapes and tensors (see `mir.proto` schema).

For list of currently supported operations, see `mir/ops/operations.lst.h`.

### How to use
Can be included as a `CMake` target.

### TODO

* Expand serialization
* Add More to readme 

### Dependencies

Mir depends on the `adtitas` library, which provides the `small_vector` data type.
