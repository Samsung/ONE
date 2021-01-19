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
#include "luci/Pass/RemoveRedundantReshapePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

class RemoveRedundantReshape : public ::testing::Test
{
public:
  RemoveRedundantReshape() {}

  void buildGraph(const std::initializer_list<uint32_t> base_shape,
                  const std::vector<int32_t> intermediate_shape,
                  const std::vector<int32_t> final_shape)
  {
    // Input Create.
    input = g.nodes()->create<luci::CircleInput>();
    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    input->shape_status(luci::ShapeStatus::VALID);
    input->rank(base_shape.size());
    input->shape(base_shape);

    // Reshape Node create.
    auto intermediate_reshape = g.nodes()->create<luci::CircleReshape>();
    intermediate_reshape->tensor(input);
    auto intermediate_const = g.nodes()->create<luci::CircleConst>();
    intermediate_const->dtype(loco::DataType::S32);
    intermediate_const->size<loco::DataType::S32>(intermediate_shape.size());
    intermediate_const->shape_status(luci::ShapeStatus::VALID);
    intermediate_const->rank(1);
    intermediate_const->dim(0).set(intermediate_shape.size());
    for (int32_t i = 0; i < intermediate_shape.size(); i++)
    {
      intermediate_const->at<loco::DataType::S32>(i) = intermediate_shape.at(i);
    }
    intermediate_reshape->shape(intermediate_const);

    // Reshape Node create.
    auto final_reshape = g.nodes()->create<luci::CircleReshape>();
    final_reshape->tensor(intermediate_reshape);
    auto final_const = g.nodes()->create<luci::CircleConst>();
    final_const->dtype(loco::DataType::S32);
    final_const->size<loco::DataType::S32>(final_shape.size());
    final_const->shape_status(luci::ShapeStatus::VALID);
    final_const->rank(1);
    final_const->dim(0).set(final_shape.size());
    for (int32_t i = 0; i < final_shape.size(); i++)
    {
      final_const->at<loco::DataType::S32>(i) = final_shape.at(i);
    }
    final_reshape->shape(final_const);

    // Output Connect.
    output = g.nodes()->create<luci::CircleOutput>();
    output->from(final_reshape);
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleOutput *output = nullptr;
};

} // namespace

TEST_F(RemoveRedundantReshape, simple_case)
{
  buildGraph({4, 6}, {-1, 4, 6}, {1, -1, 2, 3});
  luci::RemoveRedundantReshapePass pass;
  while (pass.run(&g))
    ;
  luci::CircleReshape *reshape_node = nullptr;
  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(&g)))
  {
    if (auto reshape = dynamic_cast<luci::CircleReshape *>(node))
    {
      reshape_node = reshape;
      count++;
    }
  }
  ASSERT_NE(nullptr, reshape_node);
  ASSERT_EQ(1, count);
}
