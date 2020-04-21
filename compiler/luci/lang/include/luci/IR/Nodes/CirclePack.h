/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IR_CIRCLEPACK_H__
#define __LUCI_IR_CIRCLEPACK_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/VariadicArityNode.h"

#include <cassert>

namespace luci
{

/**
 * @brief PACK in Circle
 */
class CirclePack final : public VariadicArityNode<CircleNodeImpl<CircleOpcode::PACK>>
{
public:
  CirclePack(uint32_t arity) : VariadicArityNode<CircleNodeImpl<CircleOpcode::PACK>>(arity)
  {
    // TODO Support when arity is 0
    assert(arity >= 1);
  }

public:
  uint32_t values_count(void) const { return arity(); }

public:
  Node *values(uint32_t index) const
  {
    assert(index < values_count());
    return at(index)->node();
  }
  void values(uint32_t index, Node *node)
  {
    assert(index < values_count());
    at(index)->node(node);
  }

public:
  int32_t axis(void) const { return _axis; }
  void axis(int32_t axis) { _axis = axis; }

private:
  int32_t _axis{0};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLEPACK_H__
