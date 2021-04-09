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

TEST(CloneNodeTest, clone_DepthToSpace)
{
  auto g = loco::make_graph();
  auto node_d2s = g->nodes()->create<luci::CircleDepthToSpace>();
  node_d2s->block_size(32);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_d2s, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_d2s = dynamic_cast<luci::CircleDepthToSpace *>(cloned);
  ASSERT_NE(nullptr, cloned_d2s);
  ASSERT_EQ(node_d2s->block_size(), cloned_d2s->block_size());
}
