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

#ifndef __LUCI_IR_CIRCEL_ADD_N_H__
#define __LUCI_IR_CIRCEL_ADD_N_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/VariadicArityNode.h"

namespace luci
{

/**
 * @brief ADD_N in Circle
 */
class CircleAddN final : public VariadicArityNode<CircleNodeImpl<CircleOpcode::ADD_N>>
{
public:
  CircleAddN(uint32_t arity) : VariadicArityNode<CircleNodeImpl<CircleOpcode::ADD_N>>(arity)
  {
    assert(arity >= 1);
  }

public:
  Node *inputs(uint32_t index) const { return at(index)->node(); }
  void inputs(uint32_t index, Node *node) { at(index)->node(node); }
};

} // namespace luci

#endif // __LUCI_IR_CIRCEL_ADD_N_H__
