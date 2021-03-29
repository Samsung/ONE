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
  ASSERT_EQ(nullptr, pnode.pgroup);
}

// TODO add more tests with luci::PNode

TEST(PartitionIRTest, PGroup_ctor)
{
  auto g = loco::make_graph();
  auto node = g->nodes()->create<luci::CircleAdd>();

  luci::PGroup pgroup;
  auto pnode = std::make_unique<luci::PNode>();
  pnode->node = node;

  pgroup.pnodes.push_back(std::move(pnode));

  ASSERT_NE(pgroup.pnodes.end(), pgroup.pnodes.begin());
  ASSERT_EQ(0, pgroup.inputs.size());
  ASSERT_EQ(0, pgroup.outputs.size());
}

// TODO add more tests with luci::PGroup

TEST(PartitionIRTest, PGroups_ctor)
{
  auto g = loco::make_graph();
  auto node = g->nodes()->create<luci::CircleAdd>();

  auto pnode = std::make_unique<luci::PNode>();
  pnode->node = node;

  auto pgroup = std::make_unique<luci::PGroup>();
  pgroup->pnodes.push_back(std::move(pnode));

  luci::PGroups pgroups;
  pgroups.pgroups.push_back(std::move(pgroup));

  ASSERT_NE(pgroups.pgroups.end(), pgroups.pgroups.begin());
}

// TODO add more tests with luci::PGroups
