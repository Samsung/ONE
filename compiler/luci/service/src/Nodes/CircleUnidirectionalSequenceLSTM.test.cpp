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

TEST(CloneNodeTest, clone_UnidirectionalSequenceLSTM)
{
  auto g = loco::make_graph();
  auto node_uslstm = g->nodes()->create<luci::CircleUnidirectionalSequenceLSTM>();
  node_uslstm->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_uslstm->cell_clip(1.1f);
  node_uslstm->proj_clip(2.2f);
  node_uslstm->time_major(true);
  node_uslstm->asymmetric_quantize_inputs(true);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_uslstm, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_uslstm = dynamic_cast<luci::CircleUnidirectionalSequenceLSTM *>(cloned);
  ASSERT_NE(nullptr, cloned_uslstm);
  ASSERT_EQ(node_uslstm->fusedActivationFunction(), cloned_uslstm->fusedActivationFunction());
  ASSERT_EQ(node_uslstm->cell_clip(), cloned_uslstm->cell_clip());
  ASSERT_EQ(node_uslstm->proj_clip(), cloned_uslstm->proj_clip());
  ASSERT_EQ(node_uslstm->time_major(), cloned_uslstm->time_major());
  ASSERT_EQ(node_uslstm->asymmetric_quantize_inputs(), cloned_uslstm->asymmetric_quantize_inputs());
}

TEST(CloneNodeTest, clone_UnidirectionalSequenceLSTM_NEG)
{
  auto g = loco::make_graph();
  auto node_uslstm = g->nodes()->create<luci::CircleUnidirectionalSequenceLSTM>();
  node_uslstm->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_uslstm, gc.get());
  ASSERT_EQ(nullptr, cloned);
}
