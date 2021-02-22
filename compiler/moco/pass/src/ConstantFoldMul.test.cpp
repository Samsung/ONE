/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "moco/Pass/Passes/ConstantFoldMul.h"
#include "TestHelper.h"

#include <moco/IR/TFNodes.h>
#include <loco.h>

#include <memory>

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

moco::TFConst *const_vector_init(loco::Graph *graph, std::vector<int32_t> values)
{
  auto const_node = graph->nodes()->create<moco::TFConst>();
  auto dim = values.size();

  const_node->dtype(loco::DataType::S32);
  const_node->rank(1);
  const_node->dim(0).set(dim);

  const_node->size<loco::DataType::S32>(dim);
  for (int32_t i = 0; i < dim; ++i)
    const_node->at<loco::DataType::S32>(i) = values[i];

  return const_node;
}

} // namespace

TEST(ConstantFoldMul, basic_vector)
{
  loco::Graph graph;

  auto mul_node = graph.nodes()->create<moco::TFMul>();
  {
    auto const_from_ss = const_vector_init(&graph, {1, 3, 5});
    mul_node->x(const_from_ss);

    auto const_y = const_vector_init(&graph, {2});
    mul_node->y(const_y);
  }
  setup_output_node(&graph, mul_node);

  auto pass = std::make_unique<moco::ConstantFoldMul>();
  bool cont = true;
  while (cont)
  {
    cont = pass->run(&graph);
  }

  auto ssnode = find_first_node_bytype<moco::TFMul>(&graph);
  ASSERT_EQ(ssnode, nullptr);

  auto ssconst = find_first_node_bytype<moco::TFConst>(&graph);
  ASSERT_NE(ssconst, nullptr);
  ASSERT_EQ(ssconst->size<loco::DataType::S32>(), 3);
  ASSERT_EQ(ssconst->at<loco::DataType::S32>(0), 2);
  ASSERT_EQ(ssconst->at<loco::DataType::S32>(1), 6);
  ASSERT_EQ(ssconst->at<loco::DataType::S32>(2), 10);
}

TEST(ConstantFoldMul, basic_refinedet_1)
{
  loco::Graph graph;

  auto mul_node = graph.nodes()->create<moco::TFMul>();
  {
    auto const_from_ss = const_vector_init(&graph, {5});
    mul_node->x(const_from_ss);

    auto const_y = const_vector_init(&graph, {2});
    mul_node->y(const_y);
  }
  setup_output_node(&graph, mul_node);

  auto pass = std::make_unique<moco::ConstantFoldMul>();
  bool cont = true;
  while (cont)
  {
    cont = pass->run(&graph);
  }

  auto ssnode = find_first_node_bytype<moco::TFMul>(&graph);
  ASSERT_EQ(ssnode, nullptr);

  auto ssconst = find_first_node_bytype<moco::TFConst>(&graph);
  ASSERT_NE(ssconst, nullptr);
  ASSERT_EQ(ssconst->size<loco::DataType::S32>(), 1);
  ASSERT_EQ(ssconst->at<loco::DataType::S32>(0), 10);
}
