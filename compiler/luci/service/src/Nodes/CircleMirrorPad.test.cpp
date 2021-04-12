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

TEST(CloneNodeTest, clone_MirrorPad)
{
  auto g = loco::make_graph();
  auto node_mp = g->nodes()->create<luci::CircleMirrorPad>();
  node_mp->mode(luci::MirrorPadMode::REFLECT);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_mp, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_mp = dynamic_cast<luci::CircleMirrorPad *>(cloned);
  ASSERT_NE(nullptr, cloned_mp);
  ASSERT_EQ(node_mp->mode(), cloned_mp->mode());
}

TEST(CloneNodeTest, clone_MirrorPad_mode_NEG)
{
  auto g = loco::make_graph();
  auto node_mp = g->nodes()->create<luci::CircleMirrorPad>();
  node_mp->mode(luci::MirrorPadMode::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_mp, gc.get());
  ASSERT_EQ(nullptr, cloned);
}
