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

TEST(CloneNodeTest, clone_FakeQuant)
{
  auto g = loco::make_graph();
  auto node_fq = g->nodes()->create<luci::CircleFakeQuant>();
  node_fq->min(1.0f);
  node_fq->max(2.0f);
  node_fq->num_bits(8);
  node_fq->narrow_range(true);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_fq, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_fq = dynamic_cast<luci::CircleFakeQuant *>(cloned);
  ASSERT_NE(nullptr, cloned_fq);
  ASSERT_EQ(node_fq->min(), cloned_fq->min());
  ASSERT_EQ(node_fq->max(), cloned_fq->max());
  ASSERT_EQ(node_fq->num_bits(), cloned_fq->num_bits());
  ASSERT_EQ(node_fq->narrow_range(), cloned_fq->narrow_range());
}
