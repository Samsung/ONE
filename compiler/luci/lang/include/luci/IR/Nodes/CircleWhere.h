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

#ifndef __LUCI_IR_CIRCLE_WHERE_H__
#define __LUCI_IR_CIRCLE_WHERE_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/VariadicArityNode.h"

#include <cassert>

namespace luci
{

/**
 * @brief WHERE in Circle
 */
class CircleWhere final : public VariadicArityNode<CircleNodeImpl<CircleOpcode::WHERE>>
{
public:
  CircleWhere(bool provide_xy)
      : VariadicArityNode<CircleNodeImpl<CircleOpcode::WHERE>>(provide_xy ? 3 : 1)
  {
    // NOTHING TO DO
  }

public:
  loco::Node *cond() const { return at(0)->node(); }
  void cond(loco::Node *node) { at(0)->node(node); }

  loco::Node *x() const
  {
    if (arity() == 3)
      return at(1)->node();
    return nullptr;
  }
  void x(Node *node) { at(1)->node(node); }

  loco::Node *y() const
  {
    if (arity() == 3)
      return at(2)->node();
    return nullptr;
  }
  void y(Node *node) { at(2)->node(node); }
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_WHERE_H__
