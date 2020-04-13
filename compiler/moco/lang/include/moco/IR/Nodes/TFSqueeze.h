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

#ifndef __MOCO_IR_TFSQUEEZE_H__
#define __MOCO_IR_TFSQUEEZE_H__

#include "moco/IR/TFNodeDecl.h"

#include <vector>

namespace moco
{

/// @note TFSqueeze corresponds to the following GraphDef
/*
node {
  name: "squeeze"
  op: "Squeeze"
  input: "x"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "squeeze_dims"
    value {
      list {
        i: a
        i: b
        ..
      }
    }
  }
}
*/

class TFSqueeze final : public FixedArityNode<1, TFNodeImpl<TFOpcode::Squeeze>>
{
public:
  TFSqueeze() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

public:
  const std::vector<int64_t> &squeeze_dims(void) const { return _squeeze_dims; }
  void squeeze_dims(const std::vector<int64_t> &squeeze_dims) { _squeeze_dims = squeeze_dims; }

private:
  std::vector<int64_t> _squeeze_dims;
};

} // namespace moco

#endif // __MOCO_IR_TFSQUEEZE_H__
