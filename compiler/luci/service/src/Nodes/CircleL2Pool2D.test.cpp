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

#include "luci/Service/CircleNodeClone.h"

#include <gtest/gtest.h>

TEST(CloneNodeTest, clone_L2Pool2D)
{
  auto g = loco::make_graph();
  auto node_l2n = g->nodes()->create<luci::CircleL2Pool2D>();
  node_l2n->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_l2n->padding(luci::Padding::SAME);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_l2n, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_l2n = dynamic_cast<luci::CircleL2Pool2D *>(cloned);
  ASSERT_NE(nullptr, cloned_l2n);
  ASSERT_EQ(node_l2n->fusedActivationFunction(), cloned_l2n->fusedActivationFunction());
  ASSERT_EQ(node_l2n->padding(), cloned_l2n->padding());
}

TEST(CloneNodeTest, clone_L2Normalize_fusedact_NEG)
{
  auto g = loco::make_graph();
  auto node_l2n = g->nodes()->create<luci::CircleL2Pool2D>();
  node_l2n->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node_l2n->padding(luci::Padding::SAME);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_l2n, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(CloneNodeTest, clone_L2Normalize_padding_NEG)
{
  auto g = loco::make_graph();
  auto node_l2n = g->nodes()->create<luci::CircleL2Pool2D>();
  node_l2n->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_l2n->padding(luci::Padding::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_l2n, gc.get());
  ASSERT_EQ(nullptr, cloned);
}
