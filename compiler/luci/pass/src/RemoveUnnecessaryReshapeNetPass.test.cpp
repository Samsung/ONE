/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "luci/Pass/RemoveUnnecessaryReshapeNetPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

class RemoveUnnecessaryReshapeNet : public ::testing::Test
{
public:
  RemoveUnnecessaryReshapeNet() {}

  void createReshapeConst(luci::CircleReshape *target, const std::vector<uint32_t> shape)
  {
    auto shape_const = g.nodes()->create<luci::CircleConst>();
    shape_const->dtype(loco::DataType::S32);
    shape_const->size<loco::DataType::S32>(shape.size());
    shape_const->shape_status(luci::ShapeStatus::VALID);
    shape_const->rank(1);
    shape_const->dim(0).set(shape.size());
    for (int32_t i = 0; i < shape.size(); i++)
    {
      shape_const->at<loco::DataType::S32>(i) = static_cast<int32_t>(shape.at(i));
    }
    shape_const->name("shape_const");
    target->shape(shape_const);
    target->rank(shape.size());
    for (uint32_t i = 0; i < shape.size(); i++)
    {
      target->dim(i) = shape[i];
    }
    target->shape_status(luci::ShapeStatus::VALID);
  }

  void buildGraph(const std::initializer_list<uint32_t> base_shape,
                  const std::initializer_list<uint32_t> first_shape,
                  const std::initializer_list<uint32_t> second_shape)
  {
    // Input Create.
    input = g.nodes()->create<luci::CircleInput>();
    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    input->shape_status(luci::ShapeStatus::VALID);
    input->shape(base_shape);
    input->name("input");

    // Create first reshape.
    first_reshape = g.nodes()->create<luci::CircleReshape>();
    first_reshape->tensor(input);
    first_reshape->name("Reshape");
    createReshapeConst(first_reshape, first_shape);

    // Create logistic.
    logistic = g.nodes()->create<luci::CircleLogistic>();
    logistic->x(first_reshape);
    logistic->name("logistic");
    logistic->shape(first_shape);
    logistic->shape_status(luci::ShapeStatus::VALID);

    // Create second reshape.
    second_reshape = g.nodes()->create<luci::CircleReshape>();
    second_reshape->tensor(logistic);
    second_reshape->name("second_reshape");
    createReshapeConst(second_reshape, second_shape);

    // Output Connect.
    output = g.nodes()->create<luci::CircleOutput>();
    output->from(second_reshape);
    output->name("output");
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleReshape *first_reshape = nullptr;
  luci::CircleLogistic *logistic = nullptr;
  luci::CircleReshape *second_reshape = nullptr;
  luci::CircleOutput *output = nullptr;
};

} // namespace

TEST_F(RemoveUnnecessaryReshapeNet, simple_case)
{
  buildGraph({1, 1, 1, 32}, {1, 1, 32, 1}, {1, 1, 1, 32});
  luci::RemoveUnnecessaryReshapeNetPass pass;

  ASSERT_TRUE(pass.run(&g));

  int count = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(&g)))
  {
    if (auto reshape = dynamic_cast<luci::CircleReshape *>(node))
      count++;
  }
  ASSERT_EQ(0, count);
}

TEST_F(RemoveUnnecessaryReshapeNet, shape_mismatch_NEG)
{
  buildGraph({1, 1, 1, 32}, {1, 1, 32, 1}, {1, 1, 2, 16});
  luci::RemoveUnnecessaryReshapeNetPass pass;
  ASSERT_FALSE(pass.run(&g));
}
