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

#include "GraphBuilder.h"

#include "loco/IR/Nodes.h"
#include "loco/IR/CanonicalDialect.h"
#include "loco/IR/CanonicalOpcode.h"

#include <gtest/gtest.h>

TEST(GraphBuilderTest, Usecase_000)
{
  struct SampleLayer final
  {
    loco::Node *operator()(GraphBuilder::Context *ctx)
    {
      auto node = ctx->graph()->nodes()->create<loco::ConstGen>();
      ctx->stack()->push(node);
      return node;
    }
  };

  auto g = loco::make_graph();
  auto gbuilder = make_graph_builder(g.get());

  gbuilder->push<SampleLayer>();

  auto node = gbuilder->pop();

  ASSERT_EQ(1, g->nodes()->size());
  ASSERT_EQ(loco::CanonicalDialect::get(), node->dialect());
  ASSERT_EQ(static_cast<uint32_t>(loco::CanonicalOpcode::ConstGen), node->opnum());
}
