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

#include "luci/Service/CircleNodeClone.h"

// NOTE any node will do for testing
#include <luci/IR/Nodes/CircleAdd.h>

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

  auto sparam = std::make_unique<luci::SparsityParam>();
  sparam->traversal_order = {0};
  sparam->block_map = {0};
  sparam->dim_metadata = {luci::DimMetaData(luci::DimensionType::DENSE, 1)};
  node->sparsityparam(std::move(sparam));

  node->op_version(2);

  return node;
}

} // namespace

TEST(CircleNodeCloneTest, copy_attribites)
{
  auto g = loco::make_graph();
  auto node = build_simple_add_graph(g.get());

  auto copy = g->nodes()->create<luci::CircleAdd>();
  luci::copy_common_attributes(node, copy);

  ASSERT_EQ(node->name(), copy->name());
  ASSERT_EQ(node->dtype(), copy->dtype());
  ASSERT_EQ(node->rank(), copy->rank());
  ASSERT_EQ(node->shape_status(), copy->shape_status());

  const auto *qparam_node = node->quantparam();
  const auto *qparam_copy = copy->quantparam();
  ASSERT_EQ(qparam_node->scale, qparam_copy->scale);

  const auto *sparsity_node = node->sparsityparam();
  const auto *sparsity_copy = copy->sparsityparam();
  ASSERT_EQ(sparsity_node->traversal_order, sparsity_copy->traversal_order);

  ASSERT_EQ(node->op_version(), copy->op_version());
}

TEST(CircleNodeCloneTest, clone_add_node)
{
  auto g = loco::make_graph();
  auto node = build_simple_add_graph(g.get());

  auto cg = loco::make_graph();
  auto clone = clone_node(node, cg.get());

  ASSERT_NE(nullptr, clone);
  ASSERT_EQ(cg.get(), clone->graph());
  ASSERT_EQ(node->name(), clone->name());
  ASSERT_EQ(node->dtype(), clone->dtype());
  ASSERT_EQ(node->rank(), clone->rank());
  ASSERT_EQ(node->shape_status(), clone->shape_status());
}
