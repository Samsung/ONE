/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/DynamicBatchToSingleBatchPass.h"

#include <loco.h>

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

std::unique_ptr<loco::TensorShape> make_tshape(std::initializer_list<uint32_t> dims)
{
  auto tensor_shape = std::make_unique<loco::TensorShape>();
  {
    tensor_shape->rank(dims.size());
    uint32_t axis = 0;
    for (auto it = dims.begin(); it != dims.end(); ++it)
    {
      tensor_shape->dim(axis++) = *it;
    }
  }

  return std::move(tensor_shape);
}

} // namespace

TEST(DynamicBatchToSingleBatchPassTest, simple)
{
  luci::DynamicBatchToSingleBatchPass pass;

  auto g = loco::make_graph();

  auto graph_input = g->inputs()->create();
  {
    auto tensor_shape = make_tshape({1, 5, 5, 3});
    tensor_shape->dim(0).unset();
    graph_input->shape(std::move(tensor_shape));
  }

  // Create nodes to make relu traversed first
  auto input = g->nodes()->create<luci::CircleInput>();
  {
    input->index(0);
    input->shape({1, 5, 5, 3});
    input->dim(0).unset();
  }

  EXPECT_FALSE(graph_input->shape()->dim(0).known());
  EXPECT_FALSE(input->dim(0).known());

  EXPECT_TRUE(pass.run(g.get()));

  // Check input is knwon
  EXPECT_TRUE(graph_input->shape()->dim(0).known());
  EXPECT_EQ(1, graph_input->shape()->dim(0));
  EXPECT_TRUE(input->dim(0).known());
  EXPECT_EQ(1, input->dim(0));
}

TEST(DynamicBatchToSingleBatchPassTest, simple_NEG)
{
  luci::DynamicBatchToSingleBatchPass pass;

  auto g = loco::make_graph();

  auto graph_input = g->inputs()->create();
  {
    graph_input->shape({1, 5, 5, 3});
  }

  // Create nodes to make relu traversed first
  auto input = g->nodes()->create<luci::CircleInput>();
  {
    input->index(0);
    input->shape({1, 5, 5, 3});
  }

  EXPECT_FALSE(pass.run(g.get()));
}

// Remove this test if we support rank 1 in this pass
TEST(DynamicBatchToSingleBatchPassTest, rank1_NEG)
{
  luci::DynamicBatchToSingleBatchPass pass;

  auto g = loco::make_graph();

  auto graph_input = g->inputs()->create();
  {
    auto tensor_shape = make_tshape({1});
    tensor_shape->dim(0).unset();
    graph_input->shape(std::move(tensor_shape));
  }

  // Create nodes to make relu traversed first
  auto input = g->nodes()->create<luci::CircleInput>();
  {
    input->index(0);
    input->shape({1});
    input->dim(0).unset();
  }

  EXPECT_FALSE(graph_input->shape()->dim(0).known());
  EXPECT_FALSE(input->dim(0).known());

  // Rank 1 is unsupported for now
  EXPECT_ANY_THROW(pass.run(g.get()));
}
