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
#include "luci/Pass/RemoveRedundantTransposePass.h"

#include <luci/IR/CircleNodes.h>

#include <vector>

#include <gtest/gtest.h>

namespace
{

void setValue(luci::CircleConst *node, const std::vector<int> &v)
{
  node->dtype(loco::DataType::S32);
  node->size<loco::DataType::S32>(v.size());
  node->rank(1);
  node->dim(0).set(v.size());
  for (int i = 0; i < v.size(); ++i)
  {
    node->at<loco::DataType::S32>(i) = v[i];
  }
}

/**
 *  Type1
 *  BEFORE
 *         |
 *   [CircleNode]     [CircleConst]
 *           \              /
 *           [CircleTranspose]  [CircleConst]
 *                   \              /
 *                   [CircleTranspose]
 *                           |
 *
 *  AFTER
 *         |
 *   [CircleNode]
 *         |   Remove Both
 *
 * --------------------------------------------
 *
 *  Type2
 *  BEFORE
 *         |
 *   [CircleNode]     [CircleConst]
 *           \              /
 *           [CircleTranspose]  [CircleConst]
 *                   \               /
 *                   [CircleTranspose]
 *                           |
 *
 *  AFTER
 *          |                 |
 *    [CircleNode]      [CircleConst]
 *           \               /
 *           [CircleTranspose]
 *                   |
 *
 */
void create_redundunt_transpose(loco::Graph *g, const std::vector<int32_t> &perm1,
                                const std::vector<int32_t> &perm2)
{
  assert(g);

  auto input = g->nodes()->create<luci::CircleInput>();
  auto graph_input = g->inputs()->create();
  input->index(graph_input->index());

  // Create perm1
  auto perm1_node = g->nodes()->create<luci::CircleConst>();
  setValue(perm1_node, perm1);

  auto transpose1 = g->nodes()->create<luci::CircleTranspose>();
  transpose1->dtype(loco::DataType::FLOAT32);
  transpose1->a(input);
  transpose1->perm(perm1_node);

  // Create perm2
  auto perm2_node = g->nodes()->create<luci::CircleConst>();
  setValue(perm2_node, perm2);

  auto transpose2 = g->nodes()->create<luci::CircleTranspose>();
  transpose2->dtype(loco::DataType::FLOAT32);
  transpose2->a(transpose1);
  transpose2->perm(perm2_node);

  // Output
  auto output = g->nodes()->create<luci::CircleOutput>();
  output->from(transpose2);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
}

} // namespace

TEST(RemoveRedundantTransposePass, remove_consecutive_transpose_function_type1)
{
  auto graph = loco::make_graph();
  create_redundunt_transpose(graph.get(), {1, 0, 2, 3}, {1, 0, 2, 3});

  luci::RemoveRedundantTransposePass pass;
  while (pass.run(graph.get()))
    ;
  luci::CircleTranspose *transpose_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto trans = dynamic_cast<luci::CircleTranspose *>(node);
    if (not trans)
      continue;
    transpose_node = trans;
    break;
  }
  // No transpose node is in graph.
  ASSERT_EQ(nullptr, transpose_node);
}

TEST(RemoveRedundantTransposePass, remove_consecutive_transpose_function_type2)
{
  auto graph = loco::make_graph();
  create_redundunt_transpose(graph.get(), {0, 1, 3, 2}, {1, 0, 2, 3});

  luci::RemoveRedundantTransposePass pass;
  while (pass.run(graph.get()))
    ;
  luci::CircleTranspose *transpose_node = nullptr;
  for (auto node : loco::active_nodes(loco::output_nodes(graph.get())))
  {
    auto trans = dynamic_cast<luci::CircleTranspose *>(node);
    if (not trans)
      continue;
    transpose_node = trans;
    break;
  }
  // Just one transpose node, with updated perm constant.
  ASSERT_NE(nullptr, transpose_node);
  auto perm = loco::must_cast<luci::CircleConst *>(transpose_node->perm());
  ASSERT_EQ(1, perm->at<loco::DataType::S32>(0));
  ASSERT_EQ(0, perm->at<loco::DataType::S32>(1));
  ASSERT_EQ(3, perm->at<loco::DataType::S32>(2));
  ASSERT_EQ(2, perm->at<loco::DataType::S32>(3));
}
