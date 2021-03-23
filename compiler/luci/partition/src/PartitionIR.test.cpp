/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PartitionIR.h"

// NOTE any node will do for testing
#include <luci/IR/Nodes/CircleAdd.h>

#include <gtest/gtest.h>

#include <memory>

TEST(PartitionIRTest, PNode_ctor)
{
  auto g = loco::make_graph();
  auto node = g->nodes()->create<luci::CircleAdd>();

  luci::PNode pnode;
  pnode.node = node;

  ASSERT_NE(nullptr, pnode.node);
  ASSERT_EQ(nullptr, pnode.pgraph);
}

// TODO add more tests with luci::PNode

TEST(PartitionIRTest, PGraph_ctor)
{
  auto g = loco::make_graph();
  auto node = g->nodes()->create<luci::CircleAdd>();

  luci::PGraph pgraph;
  auto pnode = std::make_unique<luci::PNode>();
  pnode->node = node;

  pgraph.pnodes.push_back(std::move(pnode));

  ASSERT_NE(pgraph.pnodes.end(), pgraph.pnodes.begin());
  ASSERT_EQ(0, pgraph.inputs.size());
  ASSERT_EQ(0, pgraph.outputs.size());
}

// TODO add more tests with luci::PGraph

TEST(PartitionIRTest, PGraphs_ctor)
{
  auto g = loco::make_graph();
  auto node = g->nodes()->create<luci::CircleAdd>();

  auto pnode = std::make_unique<luci::PNode>();
  pnode->node = node;

  auto pgraph = std::make_unique<luci::PGraph>();
  pgraph->pnodes.push_back(std::move(pnode));

  luci::PGraphs pgraphs;
  pgraphs.pgraphs.push_back(std::move(pgraph));

  ASSERT_NE(pgraphs.pgraphs.end(), pgraphs.pgraphs.begin());
}

// TODO add more tests with luci::PGraphs
