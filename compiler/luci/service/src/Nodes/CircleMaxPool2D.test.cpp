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

TEST(CloneNodeTest, clone_MaxPool2D)
{
  auto g = loco::make_graph();
  auto node_mp = g->nodes()->create<luci::CircleMaxPool2D>();
  node_mp->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_mp->padding(luci::Padding::SAME);
  node_mp->filter()->h(1);
  node_mp->filter()->w(2);
  node_mp->stride()->h(3);
  node_mp->stride()->w(4);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_mp, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_mp = dynamic_cast<luci::CircleMaxPool2D *>(cloned);
  ASSERT_NE(nullptr, cloned_mp);
  ASSERT_EQ(node_mp->fusedActivationFunction(), cloned_mp->fusedActivationFunction());
  ASSERT_EQ(node_mp->padding(), cloned_mp->padding());
  ASSERT_EQ(node_mp->filter()->h(), cloned_mp->filter()->h());
  ASSERT_EQ(node_mp->filter()->w(), cloned_mp->filter()->w());
  ASSERT_EQ(node_mp->stride()->h(), cloned_mp->stride()->h());
  ASSERT_EQ(node_mp->stride()->w(), cloned_mp->stride()->w());
}

TEST(CloneNodeTest, clone_MaxPool2D_fusedact_NEG)
{
  auto g = loco::make_graph();
  auto node_mp = g->nodes()->create<luci::CircleMaxPool2D>();
  node_mp->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node_mp->padding(luci::Padding::SAME);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_mp, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(CloneNodeTest, clone_MaxPool2D_padding_NEG)
{
  auto g = loco::make_graph();
  auto node_mp = g->nodes()->create<luci::CircleMaxPool2D>();
  node_mp->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_mp->padding(luci::Padding::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_mp, gc.get());
  ASSERT_EQ(nullptr, cloned);
}
