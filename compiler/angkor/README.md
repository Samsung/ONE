# angkor

## Purpose

_angkor_ is an `nncc` core library

## How to use

_angkor_ implements abstract data type (ADT) for feature, kernel, tensor.
There are layout, shape information and enumerator and so on.

To use some of these things, just insert `include`!
```cpp
#include <nncc/core/ADT/feature/WHAT_YOU_WANT>
#include <nncc/core/ADT/kernel/WHAT_YOU_WANT>
#include <nncc/core/ADT/tensor/WHAT_YOU_WANT>
```

## Example

- `compiler/coco/core/CMakeLists.txt`

```cmake
target_link_libraries(coco_core PUBLIC angkor)
```

- `compiler/coco/core/src/IR/Arg.cpp`

```cpp
#include "coco/IR/Arg.h"

#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>

namespace
{
const nncc::core::ADT::tensor::LexicalLayout l;
}

namespace coco
{

Arg::Arg(const nncc::core::ADT::tensor::Shape &shape) : _shape{shape}, _bag{nullptr}
{
  _map.resize(nncc::core::ADT::tensor::num_elements(shape));
}

// ....

}
```
