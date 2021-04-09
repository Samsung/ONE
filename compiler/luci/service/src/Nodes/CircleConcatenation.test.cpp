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

TEST(CloneNodeTest, clone_Concatenation)
{
  auto g = loco::make_graph();
  auto node_concat = g->nodes()->create<luci::CircleConcatenation>(3);
  node_concat->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_concat->axis(7);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_concat, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_concat = dynamic_cast<luci::CircleConcatenation *>(cloned);
  ASSERT_NE(nullptr, cloned_concat);
  ASSERT_EQ(node_concat->numValues(), cloned_concat->numValues());
  ASSERT_EQ(node_concat->fusedActivationFunction(), cloned_concat->fusedActivationFunction());
  ASSERT_EQ(node_concat->axis(), cloned_concat->axis());
}

TEST(CloneNodeTest, clone_Concatenation_NEG)
{
  auto g = loco::make_graph();
  auto node_concat = g->nodes()->create<luci::CircleConcatenation>(3);
  node_concat->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_concat, gc.get());
  ASSERT_EQ(nullptr, cloned);
}
