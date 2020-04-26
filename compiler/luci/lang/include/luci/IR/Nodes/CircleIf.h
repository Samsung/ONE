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

#ifndef __LUCI_IR_CIRCLE_IF_H__
#define __LUCI_IR_CIRCLE_IF_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/VariadicArityNode.h"

#include <cassert>

namespace luci
{

/**
 * @brief IF in Circle
 */
class CircleIf final : public VariadicArityNode<CircleNodeImpl<CircleOpcode::IF>>
{
public:
  CircleIf(uint32_t arity, uint32_t out)
      : VariadicArityNode<CircleNodeImpl<CircleOpcode::IF>>(arity + 1), _output_count(out)
  {
    assert(arity > 0);
    assert(out > 0);
  }

public:
  uint32_t input_count(void) const { return arity() - 1; }
  uint32_t output_count(void) const { return _output_count; }

public:
  Node *cond(void) const { return at(0)->node(); }
  void cond(Node *node) { at(0)->node(node); }

  Node *input(uint32_t index) const { return at(index + 1)->node(); }
  void input(uint32_t index, Node *node) { at(index + 1)->node(node); }

public:
  int32_t then_branch(void) const { return _then_branch; }
  void then_branch(int32_t then_branch) { _then_branch = then_branch; }

  int32_t else_branch(void) const { return _else_branch; }
  void else_branch(int32_t else_branch) { _else_branch = else_branch; }

public:
  loco::Graph *then_graph(void) const { return _then_graph; }
  void then_graph(loco::Graph *then_graph) { _then_graph = then_graph; }

  loco::Graph *else_graph(void) const { return _else_graph; }
  void else_graph(loco::Graph *else_graph) { _else_graph = else_graph; }

private:
  uint32_t _output_count{0};
  int32_t _then_branch{-1};
  int32_t _else_branch{-1};

  loco::Graph *_then_graph{nullptr};
  loco::Graph *_else_graph{nullptr};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_IF_H__
