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

class SubstituteTransposeToReshapeTestBase
{
public:
  SubstituteTransposeToReshapeTestBase(std::vector<int32_t> perm)
  {
    // Input Create.
    input = g.nodes()->create<luci::CircleInput>();
    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    input->shape_status(luci::ShapeStatus::VALID);
    input->rank(4);
    input->shape({126, 201, 1, 1});

    // Permutation Create.
    auto perm_const = g.nodes()->create<luci::CircleConst>();
    perm_const->dtype(loco::DataType::S32);
    perm_const->size<loco::DataType::S32>(4);
    perm_const->shape_status(luci::ShapeStatus::VALID);
    perm_const->rank(1);
    perm_const->dim(0).set(4);

    for (uint32_t i = 0; i < static_cast<uint32_t>(perm.size()); i++)
    {
      perm_const->at<loco::DataType::S32>(i) = perm.at(i);
    }

    // Transpose Create.
    auto transpose = g.nodes()->create<luci::CircleTranspose>();
    transpose->a(input);
    transpose->perm(perm_const);

    // Output Connect.
    output = g.nodes()->create<luci::CircleOutput>();
    output->from(transpose);
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleOutput *output = nullptr;
};

class SubstituteTransposeToReshapeTest : public SubstituteTransposeToReshapeTestBase,
                                         public ::testing::Test
{
public:
  SubstituteTransposeToReshapeTest()
    : SubstituteTransposeToReshapeTestBase(std::vector<int32_t>({2, 0, 3, 1}))
  {
  }
};

class SubstituteTransposeToReshapeTest_NEG : public SubstituteTransposeToReshapeTestBase,
                                             public ::testing::Test
{
public:
  SubstituteTransposeToReshapeTest_NEG()
    : SubstituteTransposeToReshapeTestBase(std::vector<int32_t>({2, 1, 3, 0}))
  {
  }
};

} // namespace

TEST_F(SubstituteTransposeToReshapeTest, simple_case)
{
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

TEST_F(SubstituteTransposeToReshapeTest_NEG, simple_case_NEG)
{
  luci::SubstituteTransposeToReshapePass pass;
  while (pass.run(&g))
    ;

  auto reshape_node = dynamic_cast<luci::CircleReshape *>(output->from());
  auto transpose_node = dynamic_cast<luci::CircleTranspose *>(output->from());
  ASSERT_EQ(nullptr, reshape_node);
  ASSERT_NE(nullptr, transpose_node);
  auto perm = loco::must_cast<luci::CircleConst *>(transpose_node->perm());
  ASSERT_EQ(2, perm->at<loco::DataType::S32>(0));
  ASSERT_EQ(1, perm->at<loco::DataType::S32>(1));
  ASSERT_EQ(3, perm->at<loco::DataType::S32>(2));
  ASSERT_EQ(0, perm->at<loco::DataType::S32>(3));
}
