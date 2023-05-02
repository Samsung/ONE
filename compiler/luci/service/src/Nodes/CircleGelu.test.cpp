/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

TEST(CloneNodeTest, clone_Gelu)
{
  auto g = loco::make_graph();
  auto node_gelu = g->nodes()->create<luci::CircleGelu>();
  node_gelu->approximate(false);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_gelu, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_gelu = dynamic_cast<luci::CircleGelu *>(cloned);
  ASSERT_NE(nullptr, cloned_gelu);
  ASSERT_EQ(node_gelu->approximate(), cloned_gelu->approximate());
}
