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

#include <luci/IR/CircleNodes.h>
#include <luci/Service/CircleShapeInference.h>

#include <loco/IR/TensorShape.h>

#include <gtest/gtest.h>

TEST(ShapeRuleTest, simple_valid_pad_avgpool2d)
{
  luci::CircleInput input;
  luci::CircleAveragePool2D avgpool_2d;

  input.shape({1, 4, 3, 1});
  input.shape_status(luci::ShapeStatus::VALID);

  avgpool_2d.value(&input);
  avgpool_2d.filter()->h(2);
  avgpool_2d.filter()->w(2);
  avgpool_2d.stride()->h(2);
  avgpool_2d.stride()->w(2);
  avgpool_2d.fusedActivationFunction(luci::FusedActFunc::NONE);
  avgpool_2d.padding(luci::Padding::VALID);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&avgpool_2d, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(2, shape.dim(1).value());
  ASSERT_EQ(1, shape.dim(2).value());
  ASSERT_EQ(1, shape.dim(3).value());
}

TEST(ShapeRuleTest, simple_same_pad_avgpool2d)
{
  luci::CircleInput input;
  luci::CircleAveragePool2D avgpool_2d;

  input.shape({1, 4, 3, 1});
  input.shape_status(luci::ShapeStatus::VALID);

  avgpool_2d.value(&input);
  avgpool_2d.filter()->h(2);
  avgpool_2d.filter()->w(2);
  avgpool_2d.stride()->h(2);
  avgpool_2d.stride()->w(2);
  avgpool_2d.fusedActivationFunction(luci::FusedActFunc::NONE);
  avgpool_2d.padding(luci::Padding::SAME);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&avgpool_2d, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(2, shape.dim(1).value());
  ASSERT_EQ(2, shape.dim(2).value());
  ASSERT_EQ(1, shape.dim(3).value());
}

TEST(CloneNodeTest, clone_AveragePool2D)
{
  auto g = loco::make_graph();
  auto node_avgpool2d = g->nodes()->create<luci::CircleAveragePool2D>();
  node_avgpool2d->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_avgpool2d->padding(luci::Padding::SAME);
  node_avgpool2d->filter()->h(1);
  node_avgpool2d->filter()->w(2);
  node_avgpool2d->stride()->h(3);
  node_avgpool2d->stride()->w(4);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_avgpool2d, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_avgpool2d = dynamic_cast<luci::CircleAveragePool2D *>(cloned);
  ASSERT_NE(nullptr, cloned_avgpool2d);
  ASSERT_EQ(node_avgpool2d->fusedActivationFunction(), cloned_avgpool2d->fusedActivationFunction());
  ASSERT_EQ(node_avgpool2d->padding(), cloned_avgpool2d->padding());
  ASSERT_EQ(node_avgpool2d->filter()->h(), cloned_avgpool2d->filter()->h());
  ASSERT_EQ(node_avgpool2d->filter()->w(), cloned_avgpool2d->filter()->w());
  ASSERT_EQ(node_avgpool2d->stride()->h(), cloned_avgpool2d->stride()->h());
  ASSERT_EQ(node_avgpool2d->stride()->w(), cloned_avgpool2d->stride()->w());
}

TEST(CloneNodeTest, clone_AveragePool2D_fusedact_NEG)
{
  auto g = loco::make_graph();
  auto node_avgpool2d = g->nodes()->create<luci::CircleAveragePool2D>();
  node_avgpool2d->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node_avgpool2d->padding(luci::Padding::SAME);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_avgpool2d, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(CloneNodeTest, clone_AveragePool2D_padding_NEG)
{
  auto g = loco::make_graph();
  auto node_avgpool2d = g->nodes()->create<luci::CircleAveragePool2D>();
  node_avgpool2d->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_avgpool2d->padding(luci::Padding::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_avgpool2d, gc.get());
  ASSERT_EQ(nullptr, cloned);
}
