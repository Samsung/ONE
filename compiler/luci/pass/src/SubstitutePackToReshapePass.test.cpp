/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "luci/Pass/SubstitutePackToReshapePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *           BEFORE
 *             |
 *        [CircleNode]
 *             |
 *        [CirclePack]
 *             |
 *        [CircleNode]
 *             |
 *
 *           AFTER
 *      |
 * [CircleNode]  [CircleConst]
 *       \             /
 *       [CircleReshape]
 *             |
 *        [CircleNode]
 *             |
 *
 */
void create_substitute_pack_to_reshape(loco::Graph *g, const std::initializer_list<uint32_t> shape,
                                       int32_t axis)
{
  assert(g);

  // Input Create.
  auto input = g->nodes()->create<luci::CircleInput>();
  auto graph_input = g->inputs()->create();
  input->index(graph_input->index());
  input->shape_status(luci::ShapeStatus::VALID);
  input->rank(shape.size());
  input->shape(shape);

  // Pack Node create.
  auto pack = g->nodes()->create<luci::CirclePack>(1);
  pack->values(0, input);
  pack->axis(axis);

  // Output Connect.
  auto output = g->nodes()->create<luci::CircleOutput>();
  output->from(pack);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());

  return;
}

} // namespace

TEST(SubstitutePackToReshapePass, simple_case)
{
  auto graph = loco::make_graph();
  create_substitute_pack_to_reshape(graph.get(), {1, 2, 3, 4}, 0);
  luci::SubstitutePackToReshapePass pass;
  while (pass.run(graph.get()))
    ;
  luci::CircleReshape *reshape_node = nullptr;
  luci::CirclePack *pack_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto reshape = dynamic_cast<luci::CircleReshape *>(node);
    if (not reshape)
    {
      auto pack = dynamic_cast<luci::CirclePack *>(node);
      if(not pack)
        continue;
      pack_node = pack;
      continue;
    }
    reshape_node = reshape;
    break;
  }
  ASSERT_NE(nullptr, reshape_node);
  ASSERT_EQ(nullptr, pack_node);
  auto new_shape = loco::must_cast<luci::CircleConst *>(reshape_node->shape());
  ASSERT_EQ(1, new_shape->at<loco::DataType::S32>(0));
  ASSERT_EQ(1, new_shape->at<loco::DataType::S32>(1));
  ASSERT_EQ(2, new_shape->at<loco::DataType::S32>(2));
  ASSERT_EQ(3, new_shape->at<loco::DataType::S32>(3));
  ASSERT_EQ(4, new_shape->at<loco::DataType::S32>(4));
}
