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

#include "luci/Profile/CircleNodeID.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

TEST(LuciCircleNodeID, simple_circle_node_id)
{
  auto g = loco::make_graph();
  auto add = g->nodes()->create<luci::CircleAdd>();

  ASSERT_FALSE(has_node_id(add));

  set_node_id(add, 3);

  ASSERT_TRUE(has_node_id(add));
  ASSERT_EQ(3, get_node_id(add));
}

TEST(LuciCircleNodeID, simple_circle_node_id_NEG)
{
  auto g = loco::make_graph();
  auto add = g->nodes()->create<luci::CircleAdd>();

  ASSERT_FALSE(has_node_id(add));

  ASSERT_ANY_THROW(get_node_id(add));
}
