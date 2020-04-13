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

#include "moco/IR/VariadicArityNode.h"

#include <loco/IR/Nodes.h>

#include <gtest/gtest.h>

namespace
{

using namespace moco;

class ArbitraryInputNode : public VariadicArityNode<loco::Node>
{
public:
  ArbitraryInputNode(uint32_t arity) : VariadicArityNode<loco::Node>(arity) {}

  void input(uint32_t idx, loco::Node *node) { at(idx)->node(node); }
  loco::Node *input(uint32_t idx) const { return at(idx)->node(); }

  const loco::Dialect *dialect(void) const { return nullptr; } // this won't be called for testing
  uint32_t opnum(void) const { return -1; }                    // this won't be called for testing
};

} // namespace

TEST(CustomOpTest, VariadicArityNode_arity_n)
{
  loco::ConstGen cg0, cg1, cg2;

  ArbitraryInputNode a_node(3);
  a_node.input(0, &cg0);
  a_node.input(1, &cg1);
  a_node.input(2, &cg2);

  ASSERT_EQ(a_node.arity(), 3);
  ASSERT_EQ(a_node.input(0), &cg0);
  ASSERT_EQ(a_node.input(1), &cg1);
  ASSERT_EQ(a_node.input(2), &cg2);
}
