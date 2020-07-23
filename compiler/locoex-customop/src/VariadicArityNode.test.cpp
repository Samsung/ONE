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

#include "locoex/VariadicArityNode.h"

#include <loco/IR/Nodes.h>

#include <gtest/gtest.h>

namespace
{
using namespace locoex;

class TestNode : public VariadicArityNode<loco::Node>
{
public:
  TestNode(uint32_t arity) : VariadicArityNode<loco::Node>(arity) {}

  void input(uint32_t idx, loco::Node *node) { at(idx)->node(node); }
  loco::Node *input(uint32_t idx) const { return at(idx)->node(); }

  const loco::Dialect *dialect(void) const { return nullptr; } // this won't be called for testing
  uint32_t opnum(void) const { return -1; }                    // this won't be called for testing
};

class ZeroInputNode : public TestNode
{
public:
  ZeroInputNode() : TestNode(0) {}
};

class BinaryInputNode : public TestNode
{
public:
  BinaryInputNode() : TestNode(2) {}
};
} // namespace

TEST(CustomOpTest, VariadicArityNode_arity_0)
{
  loco::Pull pull;

  ZeroInputNode z_node;

  ASSERT_EQ(z_node.arity(), 0);
}

TEST(CustomOpTest, VariadicArityNode_arity_2)
{
  loco::Pull pull_00, pull_01;

  BinaryInputNode b_node;
  b_node.input(0, &pull_00);
  b_node.input(1, &pull_01);

  ASSERT_EQ(b_node.arity(), 2);
  ASSERT_EQ(b_node.input(0), &pull_00);
  ASSERT_EQ(b_node.input(1), &pull_01);
}
