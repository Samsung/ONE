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

#include "moco/Import/GraphBuilderContext.h"
#include <moco/Names.h>

#include <loco.h>

#include <oops/UserExn.h>

#include <gtest/gtest.h>

TEST(GraphBuilderContext, ctor)
{
  auto graph = loco::make_graph();
  moco::NodeDefTable nodedef;
  moco::SymbolTable nodes;
  moco::UpdateQueue updates;

  moco::GraphBuilderContext context(graph.get(), &nodedef, &nodes, &updates);

  ASSERT_EQ(context.graph(), graph.get());
  ASSERT_EQ(context.nodedef(), &nodedef);
  ASSERT_EQ(context.tensor_names(), &nodes);
  ASSERT_EQ(context.updates(), &updates);
}

TEST(SymbolTable, node_name)
{
  moco::SymbolTable table;
  loco::Pull pull_node;
  moco::TensorName name("input", 0);
  moco::TensorName invalid("invalid", 0);

  table.enroll(name, &pull_node);
  ASSERT_EQ(table.node(name), &pull_node);
  // duplicate name should throw
  EXPECT_THROW(table.enroll(name, &pull_node), oops::UserExn);
  // unregistered name should throw
  EXPECT_THROW(table.node(invalid), oops::UserExn);
}

namespace
{

class TestGraphUpdate final : public moco::GraphUpdate
{
public:
  void input(const moco::SymbolTable *) const override;
};

void TestGraphUpdate::input(const moco::SymbolTable *) const {}

} // namespace

TEST(GraphUpdateQueue, queue)
{
  std::unique_ptr<TestGraphUpdate> update(new TestGraphUpdate());
  moco::UpdateQueue updates;

  updates.enroll(std::move(update));
  auto &queue = updates.queue();
  ASSERT_EQ(queue.size(), 1);
}
