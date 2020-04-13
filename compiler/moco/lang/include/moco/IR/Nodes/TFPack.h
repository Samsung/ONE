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

#ifndef __MOCO_IR_TFPACK_H__
#define __MOCO_IR_TFPACK_H__

#include "moco/IR/TFNodeDecl.h"
#include "moco/IR/VariadicArityNode.h"

namespace moco
{
/// @note TFPack corresponds to the following GraphDef
/*
node {
  name: "Pack"
  op: "Pack"
  input: "input_1"
  input: "input_2"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
*/

class TFPack final : public VariadicArityNode<TFNodeImpl<TFOpcode::Pack>>
{
public:
  TFPack(uint32_t arity) : VariadicArityNode<TFNodeImpl<TFOpcode::Pack>>(arity)
  {
    // at least one item should exist
    assert(arity >= 1);
  }

public:
  Node *values(uint32_t index) const
  {
    assert(index < arity());
    return at(index)->node();
  }
  void values(uint32_t index, Node *node)
  {
    assert(index < arity());
    at(index)->node(node);
  }

public:
  uint32_t N(void) const { return arity(); }

  int32_t axis(void) const { return _axis; }
  void axis(int32_t axis) { _axis = axis; }

private:
  int32_t _axis{0};
};

} // namespace moco

#endif // __MOCO_IR_TFPACK_H__
