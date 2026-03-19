# moco-tf

_moco-tf_ translates a TensorFlow model into _loco_

## Purpose

_moco-tf_ converts a TensorFlow generated model file to in-memory _loco_ IR Graph.

## How to use

```cxx
#include <moco/tf/Frontend.h>

...

  ::moco::tf::Frontend moco;

  std::string pb_path = "path_to_pb_file_to_load";

  auto loco_graph = moco.load(sig, pb_path, ::moco::tf::Frontend::FileType::Binary);
```

## Dependency

Please refer to [requires.cmake](./requires.cmake) for dependant modules.

## Naming rules

### TensorFlow node names

Use `REGISTER_OP` argument used in TensorFlow source `core` folder.

```
cd tensorflow/core
grep -Rn "REGISTER_OP"
```

To see single Op, `Conv2D` for example
```
cd tensorflow/core
grep -Rn "REGISTER_OP" | grep "Conv2D"
```

### Names related with TensorFlow nodes

Like `GraphBuilder` and `Canonicalization`, TensorFlow node names can be used as
prefix or suffix.

- `Conv2DGraphBuilder`
- `Conv2DCanonicalizier`

### TensorFlow Dialect IR

Use `TF` prefix with TensorFlow Dialect node names

- `TFAvgPool`
- `TFConv2D`
