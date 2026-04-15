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

#ifndef __MOCO_IR_TFCONCATV2_H__
#define __MOCO_IR_TFCONCATV2_H__

#include "moco/IR/TFNodeDecl.h"
#include "moco/IR/VariadicArityNode.h"

namespace moco
{

/// @note TFConcatV2 corresponds to the following GraphDef
/*
node {
  name: "Concat"
  op: "ConcatV2"
  input: "Input01"
  input: "Input02"
  input: "Axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
*/

class TFConcatV2 final : public VariadicArityNode<TFNodeImpl<TFOpcode::ConcatV2>>
{
public:
  TFConcatV2(uint32_t arity) : VariadicArityNode<TFNodeImpl<TFOpcode::ConcatV2>>(arity + 1)
  {
    // we add +1 for axis of VariadicArityNode ctor
    // at least one value is required
    assert(arity >= 1);
  }

public:
  uint32_t num_values(void) const
  {
    // last one is for axis
    return arity() - 1;
  }

public:
  Node *values(uint32_t index) const
  {
    assert(index < num_values());
    return at(index)->node();
  }
  void values(uint32_t index, Node *node)
  {
    assert(index < num_values());
    at(index)->node(node);
  }

  Node *axis(void) const { return at(num_values())->node(); }
  void axis(Node *node) { at(num_values())->node(node); }
};

} // namespace moco

#endif // __MOCO_IR_TFCONCATV2_H__
