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

#include "moco/Pass/Passes/ConstantFoldPack.h"
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

TEST(ConstantFoldPack, basic_scalar4_vector)
{
  loco::Graph graph;

  auto pack_node = graph.nodes()->create<moco::TFPack>(4);
  {
    auto input_0 = const_vector_init(&graph, {1});
    pack_node->values(0, input_0);

    auto input_1 = const_vector_init(&graph, {10});
    pack_node->values(1, input_1);

    auto input_2 = const_vector_init(&graph, {10});
    pack_node->values(2, input_2);

    auto input_3 = const_vector_init(&graph, {64});
    pack_node->values(3, input_3);
  }
  // add Identity node as the output Pack will be replaced
  auto identity = graph.nodes()->create<moco::TFIdentity>();
  identity->input(pack_node);
  setup_output_node(&graph, identity);

  auto pass = std::make_unique<moco::ConstantFoldPack>();
  bool cont = true;
  while (cont)
  {
    cont = pass->run(&graph);
  }

  auto pnode = find_first_node_bytype<moco::TFPack>(&graph);
  ASSERT_EQ(pnode, nullptr);

  auto pconst = find_first_node_bytype<moco::TFConst>(&graph);
  ASSERT_NE(pconst, nullptr);
  ASSERT_EQ(pconst->rank(), 2);
  ASSERT_EQ(pconst->size<loco::DataType::S32>(), 4);
  ASSERT_EQ(pconst->at<loco::DataType::S32>(0), 1);
  ASSERT_EQ(pconst->at<loco::DataType::S32>(1), 10);
  ASSERT_EQ(pconst->at<loco::DataType::S32>(2), 10);
  ASSERT_EQ(pconst->at<loco::DataType::S32>(3), 64);
}
