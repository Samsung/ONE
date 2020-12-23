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
#include "luci/Pass/SubstituteTransposeToReshapePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

void create_substitute_transpose_to_reshape(loco::Graph *g,
                                            const std::initializer_list<uint32_t> shape,
                                            const std::vector<int32_t> perm)
{
  assert(g);
  // Input Create.
  auto input = g->nodes()->create<luci::CircleInput>();
  auto graph_input = g->inputs()->create();
  input->index(graph_input->index());
  input->shape_status(luci::ShapeStatus::VALID);
  input->rank(shape.size());
  input->shape(shape);

  // Permutation Create.
  auto perm_const = g->nodes()->create<luci::CircleConst>();
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
  auto transpose_node = g->nodes()->create<luci::CircleTranspose>();
  transpose_node->a(input);
  transpose_node->perm(perm_const);

  // Output Connect.
  auto output = g->nodes()->create<luci::CircleOutput>();
  output->from(transpose_node);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
}

} // namespace

TEST(SubstituteTransposeToReshapePass, simple_case)
{
  auto graph = loco::make_graph();
  create_substitute_transpose_to_reshape(graph.get(), {126, 201, 1, 1},
                                         std::vector<int32_t>({2, 0, 3, 1}));
  luci::SubstituteTransposeToReshapePass pass;
  while (pass.run(graph.get()))
    ;
  luci::CircleReshape *reshape_node = nullptr;
  luci::CircleTranspose *transpose_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    if (auto reshape = dynamic_cast<luci::CircleReshape *>(node))
      reshape_node = reshape;
    else if (auto pack = dynamic_cast<luci::CircleTranspose *>(node))
      transpose_node = pack;
  }
  ASSERT_NE(nullptr, reshape_node);
  ASSERT_EQ(nullptr, transpose_node);
  auto new_shape = loco::must_cast<luci::CircleConst *>(reshape_node->shape());
  ASSERT_EQ(1, new_shape->at<loco::DataType::S32>(0));
  ASSERT_EQ(126, new_shape->at<loco::DataType::S32>(1));
  ASSERT_EQ(1, new_shape->at<loco::DataType::S32>(2));
  ASSERT_EQ(201, new_shape->at<loco::DataType::S32>(3));
}

TEST(SubstituteTransposeToReshapePass, simple_case_NEG)
{
  auto graph = loco::make_graph();
  create_substitute_transpose_to_reshape(graph.get(), {126, 201, 1, 1},
                                         std::vector<int32_t>({2, 1, 3, 0}));
  luci::SubstituteTransposeToReshapePass pass;
  while (pass.run(graph.get()))
    ;
  luci::CircleReshape *reshape_node = nullptr;
  luci::CircleTranspose *transpose_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    if (auto reshape = dynamic_cast<luci::CircleReshape *>(node))
      reshape_node = reshape;
    else if (auto tran = dynamic_cast<luci::CircleTranspose *>(node))
      transpose_node = tran;
  }
  ASSERT_EQ(nullptr, reshape_node);
  ASSERT_NE(nullptr, transpose_node);
  auto perm = loco::must_cast<luci::CircleConst *>(transpose_node->perm());
  ASSERT_EQ(2, perm->at<loco::DataType::S32>(0));
  ASSERT_EQ(1, perm->at<loco::DataType::S32>(1));
  ASSERT_EQ(3, perm->at<loco::DataType::S32>(2));
  ASSERT_EQ(0, perm->at<loco::DataType::S32>(3));
}
