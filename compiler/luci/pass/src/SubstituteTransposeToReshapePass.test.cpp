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
#include "luci/Pass/SubstituteTransposeToReshapePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

class SubstituteTransposeToReshapeTest : public ::testing::Test
{
public:
  SubstituteTransposeToReshapeTest() {}

  void buildGraph(const std::initializer_list<uint32_t> shape, const std::vector<int32_t> perm)
  {
    // Input Create.
    input = g.nodes()->create<luci::CircleInput>();
    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    input->shape_status(luci::ShapeStatus::VALID);
    input->rank(shape.size());
    input->shape(shape);

    // Permutation Create.
    auto perm_const = g.nodes()->create<luci::CircleConst>();
    perm_const->dtype(loco::DataType::S32);
    perm_const->size<loco::DataType::S32>(perm.size());
    perm_const->shape_status(luci::ShapeStatus::VALID);
    perm_const->rank(1);
    perm_const->dim(0).set(perm.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(perm.size()); i++)
    {
      perm_const->at<loco::DataType::S32>(i) = perm.at(i);
    }

    // Transpose Create.
    auto transpose_node = g.nodes()->create<luci::CircleTranspose>();
    transpose_node->a(input);
    transpose_node->perm(perm_const);

    // Output Connect.
    output = g.nodes()->create<luci::CircleOutput>();
    output->from(transpose_node);
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleOutput *output = nullptr;
};

} // namespace

TEST(SubstituteTransposeToReshapePassTest, name)
{
  luci::SubstituteTransposeToReshapePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(SubstituteTransposeToReshapeTest, simple_case)
{
  // Create graph that tranpose input {126, 201, 1, 1} with permutation {2, 0, 3, 1}
  buildGraph({126, 201, 1, 1}, std::vector<int32_t>({2, 0, 3, 1}));
  // With this input shape and permutation values, output shape will be [1, 126, 1, 201].
  // The order of non-one values is unchanged (126, 201).
  // So this Transpose op can be converted to Reshape op.
  luci::SubstituteTransposeToReshapePass pass;
  while (pass.run(&g))
    ;

  auto reshape_node = dynamic_cast<luci::CircleReshape *>(output->from());
  auto transpose_node = dynamic_cast<luci::CircleTranspose *>(output->from());
  ASSERT_NE(nullptr, reshape_node);
  ASSERT_EQ(nullptr, transpose_node);
  auto new_shape = loco::must_cast<luci::CircleConst *>(reshape_node->shape());
  ASSERT_EQ(1, new_shape->at<loco::DataType::S32>(0));
  ASSERT_EQ(126, new_shape->at<loco::DataType::S32>(1));
  ASSERT_EQ(1, new_shape->at<loco::DataType::S32>(2));
  ASSERT_EQ(201, new_shape->at<loco::DataType::S32>(3));
}

TEST_F(SubstituteTransposeToReshapeTest, failed_to_substitute_NEG)
{
  // Create graph that tranpose input {126, 201, 1, 1} with permutation {2, 1, 3, 0}
  buildGraph({126, 201, 1, 1}, std::vector<int32_t>({2, 1, 3, 0}));
  // With this input shape and permutation values, output shape will be [1, 201, 1, 126].
  // The order of non-one values is changed (126, 201) -> (201, 126).
  // So this Transpose op cannot be converted to Reshape op.
  luci::SubstituteTransposeToReshapePass pass;
  while (pass.run(&g))
    ;

  auto reshape_node = dynamic_cast<luci::CircleReshape *>(output->from());
  auto transpose_node = dynamic_cast<luci::CircleTranspose *>(output->from());
  ASSERT_EQ(nullptr, reshape_node);
  ASSERT_NE(nullptr, transpose_node);
}
