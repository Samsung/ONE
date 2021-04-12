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

TEST(CloneNodeTest, clone_BCQGather)
{
  auto g = loco::make_graph();
  auto node_gat = g->nodes()->create<luci::CircleBCQGather>();
  node_gat->axis(3);
  node_gat->input_hidden_size(5);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_gat, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_gat = dynamic_cast<luci::CircleBCQGather *>(cloned);
  ASSERT_NE(nullptr, cloned_gat);
  ASSERT_EQ(node_gat->axis(), cloned_gat->axis());
  ASSERT_EQ(node_gat->input_hidden_size(), cloned_gat->input_hidden_size());
}
