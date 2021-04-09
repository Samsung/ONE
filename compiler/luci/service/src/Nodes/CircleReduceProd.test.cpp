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

TEST(CloneNodeTest, clone_ReduceProd)
{
  auto g = loco::make_graph();
  auto node_rp = g->nodes()->create<luci::CircleReduceProd>();
  node_rp->keep_dims(true);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_rp, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_rp = dynamic_cast<luci::CircleReduceProd *>(cloned);
  ASSERT_NE(nullptr, cloned_rp);
  ASSERT_EQ(node_rp->keep_dims(), cloned_rp->keep_dims());
}
