# locomotiv
_locomotiv_ is a reference interpreter for _loco_ IR.

# Purpose
- _locomotiv_ would serve as code level specification and a reference implementation for loco IR.
- _locomotiv_ is required for loco-related tools to be tested.

# Sample code to use locomotiv library
This sample code shows how to use locomotiv. Please refer to `src/Session.test.cpp` as well for actual usage.
```cpp
template <typename T> using Buffer = nncc::core::ADT::tensor::Buffer<T>

loco::Graph *graph;
// ... building graph ...

// Open interpreter session
locomotiv::Session sess(graph);

for (uint32_t i = 0; i < s.input_size(); ++i)
{
  Buffer<type> buffer;
  // ... building buffer ...

  locomotiv::NodeData input_data = locomotiv::make_data(buffer);

  sess.set_input(i, input_data);
}

// Run inference
sess.infer();

// Query inferred output
locomotiv::NodeData *output_data = sess.get_output(query_index);

// Get buffer according to data type
switch(output_data->dtype())
{
case loco::DataType::S32:
{
  Buffer<int32_t> output_buffer = output_data->as_s32_bufptr();
  // Do something
  break;
}
case loco::DataType::FLOAT32:
{
  Buffer<float> output_buffer = output_data->as_f32_bufptr();
  // Do something
  break;
}
// ...
}
```

# How to support new loco node execution: recommended guide

## Steps to support new loco node
1. First of all, understand semantics of the node to newly support, especially on calculation spec and valid use cases.
2. Add the node to `locomotiv/src/Node.lst`. Please keep alphabetical order. This automatically declares `NodeExecution::execute(TheNode *)` and updates `NodeExecution::run()` to deal with the node.
3. Define `execute(loco::TheNode *)` at `locomotiv/src/Node/TheNode.cpp`.
4. Test new node execution at `locomotiv/src/Node/TheNode.test.cpp` if possible.

### Note on internal data layout rule
For each domain (see `loco::Domain`), `locomotiv` has fixed layout rule on how to store its data in memory.
- Feature is represented as NHWC layout
  - That is number of batch (N), height (H), width (W) and channel depth (C)
- Filter is represented as NHWC layout
  - That is number of filter (N), height (H), width (W) and input channel depth (C)
- DepthwiseFilter is represented as HWCM layout
  - That is height (H), width (W), input channel depth (C) and depth multiplier (M)
- Matrix is represented as HW layout
  - That is height (H), width (W)

### Notes on step 3
- Mocking Tensorflow lite `reference_op.h` might be a good place to start.
- `execute()` can be called multiple times. It just recalculates and updates annotated data. So it should `erase_annot_data()` before newly `annot_data()`.
- Most node execution behaviour would be implemented for each data type.
- `execute()` should throw runtime error on invalid cases. Some of these cases are explained:
  - Invalid argument node
    - e.g. Pull -> MaxPool2D is invalid as MaxPool2D requires feature map as its argument.
  - Lack of argument data
    - e.g. Given 'Pull -> Push' graph. On execution of Push, if no NodeData annotated to Pull, it is invalid.
  - Mismatch of argument shapes
    - e.g. Addition between 2x2 and 3x3 tensor is invalid
    - e.g. MaxPool2D expects its ifm to be 4D feature, otherwise invalid.
  - Mismatch between node's own information and inferred information
    - Some node already have attributes like shape or data type. If inferred information is different with existing node's, it is invalid.

### Recommendation on step 4 (test)
- If the node has no arguments, create a node object and `NodeExecution::run()` on it. Check whether it operates correctly.
- If the node has N (>= 1) arguments, make N pull node inputs, source them to the node to be tested. FeatureEncode or FilterEncode node may be required inbetween depending on the node's argument type. Then annotate N pull nodes with its data, `NodeExecution::run()` on the node to test, and check whether it operates correctly.
