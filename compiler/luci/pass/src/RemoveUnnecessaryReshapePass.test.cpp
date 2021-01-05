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

#include "luci/Pass/RemoveUnnecessaryReshapePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

void create_unnecessary_reshape_graph(loco::Graph *g,
                                      const std::initializer_list<uint32_t> input_shape,
                                      bool remove)
{
  assert(g);

  // Input create
  auto input = g->nodes()->create<luci::CircleInput>();
  auto graph_input = g->inputs()->create();
  input->index(graph_input->index());
  input->shape_status(luci::ShapeStatus::VALID);
  input->rank(input_shape.size());
  input->shape(input_shape);

  // Output_shape CircleConst create
  std::vector<uint32_t> shape_vector{input_shape};
  auto output_shape = g->nodes()->create<luci::CircleConst>();
  output_shape->shape_status(luci::ShapeStatus::VALID);
  output_shape->dtype(loco::DataType::S32);
  output_shape->rank(1);
  output_shape->dim(0).set(remove ? shape_vector.size() : 1);
  output_shape->size<loco::DataType::S32>(remove ? shape_vector.size() : 1);
  for (uint32_t i = 0; i < output_shape->dim(0).value(); i++)
  {
    if (remove)
      output_shape->at<loco::DataType::S32>(i) = static_cast<int32_t>(shape_vector.at(i));
    else
      output_shape->at<loco::DataType::S32>(i) = -1;
  }

  // Reshape create
  auto reshape_node = g->nodes()->create<luci::CircleReshape>();
  reshape_node->tensor(input);
  reshape_node->shape(output_shape);
  reshape_node->newShape()->rank(remove ? shape_vector.size() : 1);
  for (uint32_t i = 0; i < reshape_node->newShape()->rank(); i++)
  {
    if (remove)
      reshape_node->newShape()->dim(i) = static_cast<int32_t>(shape_vector.at(i));
    else
      reshape_node->newShape()->dim(i) = -1;
  }

  // Output create
  auto output = g->nodes()->create<luci::CircleOutput>();
  output->from(reshape_node);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
}

} // namespace

TEST(RemoveUnnecessaryReshapePass, create_unnecessary_reshape)
{
  auto graph = loco::make_graph();
  create_unnecessary_reshape_graph(graph.get(), {1, 2, 3, 4}, true);
  luci::CircleReshape *reshape_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto reshape = dynamic_cast<luci::CircleReshape *>(node);
    if (not reshape)
      continue;
    reshape_node = reshape;
    break;
  }
  ASSERT_NE(nullptr, reshape_node);
  luci::RemoveUnnecessaryReshapePass pass;
  while (pass.run(graph.get()))
    ;
  reshape_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto reshape = dynamic_cast<luci::CircleReshape *>(node);
    if (not reshape)
      continue;
    reshape_node = reshape;
    break;
  }
  ASSERT_EQ(nullptr, reshape_node);
}

TEST(RemoveUnnecessaryReshapePass, create_unnecessary_reshape_NEG)
{
  auto graph = loco::make_graph();
  create_unnecessary_reshape_graph(graph.get(), {1, 2, 3, 4}, false);
  luci::CircleReshape *reshape_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto reshape = dynamic_cast<luci::CircleReshape *>(node);
    if (not reshape)
      continue;
    reshape_node = reshape;
    break;
  }
  ASSERT_NE(nullptr, reshape_node);
  luci::RemoveUnnecessaryReshapePass pass;
  while (pass.run(graph.get()))
    ;
  reshape_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto reshape = dynamic_cast<luci::CircleReshape *>(node);
    if (not reshape)
      continue;
    reshape_node = reshape;
    break;
  }
  ASSERT_NE(nullptr, reshape_node);
}
