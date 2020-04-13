/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __MOCO_IR_TFMAXPOOL_H__
#define __MOCO_IR_TFMAXPOOL_H__

#include "moco/IR/TFNodeDecl.h"

#include <vector>

namespace moco
{

/// @note TFMaxPool corresponds to the following GraphDef
/*
node {
  name: "maxpool2d"
  op: "MaxPool"
  input: "placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1 i: 2 i: 2 i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1 i: 1 i: 1 i: 1
      }
    }
  }
}
*/

class TFMaxPool final : public FixedArityNode<1, TFNodeImpl<TFOpcode::MaxPool>>
{
public:
  TFMaxPool() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { return at(0)->node(node); }

public:
  const TFDataLayout &data_layout(void) const { return _data_layout; }
  void data_layout(const TFDataLayout &data_layout) { _data_layout = data_layout; }

  const TFPadding &padding(void) const { return _padding; }
  void padding(const TFPadding &padding) { _padding = padding; }

  const std::vector<int64_t> &ksize(void) const { return _ksize; }
  void ksize(const std::vector<int64_t> &ksize) { _ksize = ksize; }

  const std::vector<int64_t> &strides(void) const { return _strides; }
  void strides(const std::vector<int64_t> &strides) { _strides = strides; }

private:
  TFDataLayout _data_layout;
  TFPadding _padding;
  std::vector<int64_t> _ksize;
  std::vector<int64_t> _strides;
};

} // namespace moco

#endif // __MOCO_IR_TFMAXPOOL_H__
