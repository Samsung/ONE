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

#ifndef __LUCI_IR_CIRCLE_WHILE_H__
#define __LUCI_IR_CIRCLE_WHILE_H__

#include "luci/IR/CircleNodeDecl.h"
#include "luci/IR/CircleOpcode.h"

#include "luci/IR/VariadicArityNode.h"

#include <cassert>

namespace luci
{

/**
 * @brief WHILE in Circle
 */
class CircleWhile final : public VariadicArityNode<CircleNodeImpl<CircleOpcode::WHILE>>
{
public:
  CircleWhile(uint32_t arity, uint32_t out)
    : VariadicArityNode<CircleNodeImpl<CircleOpcode::WHILE>>(arity), _output_count(out)
  {
    assert(arity > 0);
    assert(out > 0);

    // input and output must have the same size
    assert(arity == out);
  }

public:
  uint32_t input_count(void) const { return arity(); }
  uint32_t output_count(void) const { return _output_count; }

public:
  Node *input(uint32_t index) const { return at(index)->node(); }
  void input(uint32_t index, Node *node) { at(index)->node(node); }

public:
  int32_t cond_branch(void) const { return _cond_branch; }
  void cond_branch(int32_t cond_branch) { _cond_branch = cond_branch; }

  int32_t body_branch(void) const { return _body_branch; }
  void body_branch(int32_t body_branch) { _body_branch = body_branch; }

public:
  loco::Graph *cond_graph(void) const { return _cond_graph; }
  void cond_graph(loco::Graph *cond_graph) { _cond_graph = cond_graph; }

  loco::Graph *body_graph(void) const { return _body_graph; }
  void body_graph(loco::Graph *body_graph) { _body_graph = body_graph; }

private:
  uint32_t _output_count{0};
  int32_t _cond_branch{-1};
  int32_t _body_branch{-1};

  loco::Graph *_cond_graph{nullptr};
  loco::Graph *_body_graph{nullptr};
};

} // namespace luci

#endif // __LUCI_IR_CIRCLE_WHILE_H__
