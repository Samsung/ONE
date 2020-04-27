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

#include "loco/IR/CanonicalNode.h"

#include <gtest/gtest.h>

TEST(CanonicalNodeTest, visitor_with_user_default_impl)
{
  struct MyVisitor final : public loco::CanonicalNodeVisitor<uint32_t>
  {
    // This visitor returns 128 if it visits a Forward node.
    uint32_t visit(const loco::Forward *) final { return 128; }

    // Otherwise, this visitor returns 256.
    uint32_t visit(const loco::Node *) final { return 256; }
  };

  loco::Forward forward;
  loco::ConstGen constgen;

  MyVisitor v;

  ASSERT_EQ(128, forward.accept(&v));
  ASSERT_EQ(256, constgen.accept(&v));
}

TEST(CanonicalNodeTest, visitor)
{
  struct CountingVisitor final : public loco::CanonicalNodeVisitor<uint32_t>
  {
    uint32_t visit(const loco::Forward *) final { return 1; }
  };

  // Visitor can visit constant nodes
  const loco::Forward node;

  CountingVisitor v;

  ASSERT_EQ(1, node.accept(&v));
}

TEST(CanonicalNodeTest, mutable_visitor)
{
  struct ResetForward final : public loco::CanonicalNodeMutableVisitor<void>
  {
    void visit(loco::Forward *node) final { node->input(nullptr); }
  };

  loco::Pull pull_node;
  loco::Forward forward_node;

  forward_node.input(&pull_node);

  ResetForward v;
  forward_node.accept(&v);

  ASSERT_EQ(nullptr, forward_node.input());
}
