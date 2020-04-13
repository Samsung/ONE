# Model

## Serialization Format

`nnpackage` uses flatbuffers to store model.

Rationale:

1. `flatbuffers` is:

- space-efficient
- explicit-schema based
- royalty-free license open-source library
- header-only solution (unless we use flatbuffer's reflection)
- proven solution (used by TensorFlow-Lite)

2. We've checked other solutions:
- [`bjson (binary JSON)`](http://bjson.org/)
- `protocol buffers`

## Baseline Schema

`nnpackage` schema is based on tensorflow-lite schema.

Rationale:

- Fundamentally, `nnpackage` and `TFLite` have same aim:
Running pre-trained models on a device, which has relatively low computing power and memory.
TFLite's solution is acceptable, we don't need to create same thing again.
- We can use several infra-structures and tools from TFLite.

## Schema Source

nnpackage supports two kinds of models: `tflite` and `circle`

- For tflite, see `schema.fbs` from tensorflow lite `v1.13.1` source.

- For circle, see [`../schema/circle_schema.fbs`](../schema/circle_schema.fbs).

## Extensions

`nnpackage` model has some extensions that are different or missing from TFLite.

### A. Multiple Layout

`nnpackage` can support multiple layouts.

1. The layout is presented using `DataFormat` enumeration.

`DataFormat` must be one of the enumeration defined in `nnpackage_schema.fbs`.

For example, `CHANNELS_FIRST` or `CHANNELS_LAST` can be used.

```
  // For 2D data, NHWC(batch, height, width, channels)
  // For 3D data, NDHWC(batch, depth, height, width, channels)
  CHANNELS_LAST = 0,
  // For 2D data, NCHW(batch, channels, height, width)
  // For 3D data, NCDHW(batch, channels, depth, height, width)
  CHANNELS_FIRST = 1,
```

2. `DataFormat` must be same within a submodel.

Rationale:

- frequent switching between different layout degrades the performance

Under this assumption, We expect to

- simplify the runtime implementation
- accelerate the performance
- reduce the memory usage

### B. Unspecified Dimension

`nnpackage` represents unspecified dimension with `-1`.

Rationale:

1. It should be `int` since dimension is int type flatbuffer schema. Thus '?' cannot be used.
2. `0` is also a candidate, which is used for Android NN API.
However, we would like to reserve `0` because `0` could be a valid dimension for a certain
operator (e.g. `tflite.slice`).

### C. Additional operators

`circle` has additional operators that are not available in `tflite`.
See operator reference below.

# Operator Reference

## TensorFlow Lite operators

All operators (except for additional operators) in `tflite` and `circle` use same semantics of tensorflow lite operators.
Refer tensorflow lite source code (our baseline: `v1.13.1`) to understand what inputs, outputs and attributes are required and how they are interpretered.

## Additional operators

### instance_norm

Applies instance normalization

y = `gamma` * (x - mean) / sqrt(variance + `epsilon`) + `beta`, where mean and variance are computed per instance per channel.

#### attributes

- `epsilon` : float (default is 1e-05)

The epsilon value is added to variance to avoid division by zero.

- `fused_activation_function` : enumeration for `fused activation function type`.

`fused activation function type` can be `NONE`, `RELU`, `RELU6` and `TANH` to name a few.
For complete list, see `circle_schema.fbs`.

The epsilon value is added to variance to avoid division by zero.

#### inputs

- `input` : 4-dimensional tensor
  - Input data tensor; dimensions are determined by `DataFormat`. See `DataFormat` for further information.
  - If `DataFormat` is `CHANNELS_FIRST`, layout will be (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.

- `gamma` : 1-dimensional tensor of size C
  - The gamma value is the scale applied to the normalized tensor

- `beta` : 1-dimensional tensor of size C
  - The beta value is the offset applied to the normalized tensor

#### outputs

- `output` : 4-dimensional tensor
  - The output tensor is the normalized tensor of the same shape and type of `input`.
