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

TEST(CloneNodeTest, clone_TransposeConv)
{
  auto g = loco::make_graph();
  auto node_trconv = g->nodes()->create<luci::CircleTransposeConv>();
  node_trconv->padding(luci::Padding::SAME);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_trconv, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_trconv = dynamic_cast<luci::CircleTransposeConv *>(cloned);
  ASSERT_NE(nullptr, cloned_trconv);
  ASSERT_EQ(node_trconv->padding(), cloned_trconv->padding());
  ASSERT_EQ(node_trconv->fusedActivationFunction(), cloned_trconv->fusedActivationFunction());
}

TEST(CloneNodeTest, clone_TransposeConv_padding_NEG)
{
  auto g = loco::make_graph();
  auto node_trconv = g->nodes()->create<luci::CircleTransposeConv>();
  node_trconv->padding(luci::Padding::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_trconv, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(CloneNodeTest, clone_TransposeConv_fAF_NEG)
{
  auto g = loco::make_graph();
  auto node_trconv = g->nodes()->create<luci::CircleTransposeConv>();
  node_trconv->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_trconv, gc.get());
  ASSERT_EQ(nullptr, cloned);
}
