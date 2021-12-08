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

// NOTE any node will do for testing
#include "luci/IR/Nodes/CircleAdd.h"

#include <loco/IR/Graph.h>

#include <gtest/gtest.h>

namespace
{

luci::CircleAdd *build_simple_add_graph(loco::Graph *g)
{
  auto node = g->nodes()->create<luci::CircleAdd>();

  node->name("name");
  node->dtype(loco::DataType::FLOAT32);
  node->rank(1);
  node->dim(0).set(3);
  node->shape_status(luci::ShapeStatus::VALID);
  node->fusedActivationFunction(luci::FusedActFunc::NONE);

  auto qparam = std::make_unique<luci::CircleQuantParam>();
  qparam->scale = {1.0};
  qparam->zerop = {0};
  qparam->min = {0.0};
  qparam->max = {1.0};
  qparam->quantized_dimension = 0;
  node->quantparam(std::move(qparam));

  return node;
}

} // namespace

TEST(CircleNodeCloneTest, copy_quantparam)
{
  auto g = loco::make_graph();
  auto node = build_simple_add_graph(g.get());

  auto copy = g->nodes()->create<luci::CircleAdd>();
  luci::copy_quantparam(node, copy);

  const auto *qparam_node = node->quantparam();
  const auto *qparam_copy = copy->quantparam();
  ASSERT_EQ(qparam_node->scale, qparam_copy->scale);
}

TEST(CircleNodeCloneTest, copy_quantparam_NEG)
{
  auto g = loco::make_graph();
  auto node = build_simple_add_graph(g.get());

  node->quantparam(nullptr);

  auto copy = g->nodes()->create<luci::CircleAdd>();
  luci::copy_quantparam(node, copy);

  const auto *qparam_copy = copy->quantparam();
  ASSERT_EQ(qparam_copy, nullptr);
}
