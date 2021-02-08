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

#include <logo/ConstantFoldingPass.h>

#include "TestHelper.h"

#include <loco.h>

#include <gtest/gtest.h>

using namespace logo::test;

TEST(ConstantFoldingTest, name)
{
  logo::ConstantFoldingPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(ConstantFoldingTest, run_NEG)
{
  loco::Graph g;
  logo::ConstantFoldingPass pass;

  ASSERT_FALSE(pass.run(&g));
}

namespace
{

/*
  test case:
      ConstGen ---- Relu ---- Push
   (-3.14, 3.14)      (0, 3.14)

  after constant folding:
                 ConstGen ------Push
                      (0, 3.14)
*/
void create_net_const_relu(loco::Graph *graph)
{
  assert(graph);

  auto const_node = graph->nodes()->create<loco::ConstGen>();
  {
    const_node->dtype(loco::DataType::FLOAT32);
    const_node->rank(1);
    const_node->dim(0) = 2;
    const_node->size<loco::DataType::FLOAT32>(2);
    const_node->at<loco::DataType::FLOAT32>(0) = -3.14f;
    const_node->at<loco::DataType::FLOAT32>(1) = 3.14f;
  }

  auto relu_node = graph->nodes()->create<loco::ReLU>();
  {
    relu_node->input(const_node);
  }

  auto push_node = graph->nodes()->create<loco::Push>();
  {
    push_node->from(relu_node);
  }

  auto graph_output = graph->outputs()->create();
  {
    graph_output->name("output");
    graph_output->dtype(loco::DataType::FLOAT32);
    loco::link(graph_output, push_node);
  }
}

} // namespace

TEST(ConstantFolding, const_relu_to_const)
{
  auto graph = loco::make_graph();
  create_net_const_relu(graph.get());

  logo::ConstantFoldingPass pass;
  while (pass.run(graph.get()) == true)
  {
    ;
  }

  auto push = logo::test::find_first_node_by_type<loco::Push>(graph.get());
  auto const_gen = loco::must_cast<loco::ConstGen *>(push->from());
  ASSERT_NE(const_gen, nullptr);

  ASSERT_EQ(const_gen->size<loco::DataType::FLOAT32>(), 2);
  ASSERT_EQ(const_gen->at<loco::DataType::FLOAT32>(0), 0); // result of relu(-3.14)
  ASSERT_EQ(const_gen->at<loco::DataType::FLOAT32>(1), 3.14f);
}

namespace
{

/*
  test case:
        ConstGen ---- Relu ---+
        (-1, 1)        (0, 1) |
                  ConstGen ---+-- ConcatV2 ----- Push
                  (2, 3)      |       (0, 1, 2, 3)
                   axis(0) ---+

  after constant folding:
                                  ConstGen ----- Push
                                  (0, 1, 2, 3)
*/
void create_net_const_relu_concat(loco::Graph *graph)
{
  assert(graph);

  auto const_1_node = graph->nodes()->create<loco::ConstGen>();
  {
    const_1_node->dtype(loco::DataType::FLOAT32);
    const_1_node->rank(1);
    const_1_node->dim(0) = 2;
    const_1_node->size<loco::DataType::FLOAT32>(2);
    const_1_node->at<loco::DataType::FLOAT32>(0) = -1.0f;
    const_1_node->at<loco::DataType::FLOAT32>(1) = 1.0f;
  }

  auto relu_node = graph->nodes()->create<loco::ReLU>();
  {
    relu_node->input(const_1_node);
  }

  auto const_2_node = graph->nodes()->create<loco::ConstGen>();
  {
    const_2_node->dtype(loco::DataType::FLOAT32);
    const_2_node->rank(1);
    const_2_node->dim(0) = 2;
    const_2_node->size<loco::DataType::FLOAT32>(2);
    const_2_node->at<loco::DataType::FLOAT32>(0) = 2.0f;
    const_2_node->at<loco::DataType::FLOAT32>(1) = 3.0f;
  }

  auto concat_node = graph->nodes()->create<loco::TensorConcat>();
  {
    concat_node->lhs(relu_node);
    concat_node->rhs(const_2_node);
    concat_node->axis(0);
  }

  auto push_node = graph->nodes()->create<loco::Push>();
  {
    push_node->from(concat_node);
  }

  auto graph_output = graph->outputs()->create();
  {
    graph_output->name("output");
    graph_output->dtype(loco::DataType::FLOAT32);
    loco::link(graph_output, push_node);
  }
}

} // namespace

TEST(ConstantFolding, const_relu_to_concat)
{
  auto graph = loco::make_graph();
  create_net_const_relu_concat(graph.get());

  logo::ConstantFoldingPass pass;
  while (pass.run(graph.get()) == true)
  {
    ;
  }

  auto push = logo::test::find_first_node_by_type<loco::Push>(graph.get());
  auto const_gen = loco::must_cast<loco::ConstGen *>(push->from());
  ASSERT_NE(const_gen, nullptr);

  ASSERT_EQ(const_gen->size<loco::DataType::FLOAT32>(), 4);
  ASSERT_EQ(const_gen->at<loco::DataType::FLOAT32>(0), 0);
  ASSERT_EQ(const_gen->at<loco::DataType::FLOAT32>(1), 1);
  ASSERT_EQ(const_gen->at<loco::DataType::FLOAT32>(2), 2);
  ASSERT_EQ(const_gen->at<loco::DataType::FLOAT32>(3), 3);
}
