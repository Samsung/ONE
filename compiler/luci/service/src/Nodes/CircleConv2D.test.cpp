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
#include "luci/Service/CircleShapeInference.h"

#include <gtest/gtest.h>

TEST(CloneNodeTest, clone_Conv2D)
{
  auto g = loco::make_graph();
  auto node_conv2d = g->nodes()->create<luci::CircleConv2D>();
  node_conv2d->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_conv2d->padding(luci::Padding::SAME);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_conv2d, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_conv2d = dynamic_cast<luci::CircleConv2D *>(cloned);
  ASSERT_NE(nullptr, cloned_conv2d);
  ASSERT_EQ(node_conv2d->fusedActivationFunction(), cloned_conv2d->fusedActivationFunction());
  ASSERT_EQ(node_conv2d->padding(), cloned_conv2d->padding());
}

TEST(CloneNodeTest, clone_Conv2D_fusedact_NEG)
{
  auto g = loco::make_graph();
  auto node_conv2d = g->nodes()->create<luci::CircleConv2D>();
  node_conv2d->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node_conv2d->padding(luci::Padding::SAME);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_conv2d, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(CloneNodeTest, clone_Conv2D_padding_NEG)
{
  auto g = loco::make_graph();
  auto node_conv2d = g->nodes()->create<luci::CircleConv2D>();
  node_conv2d->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_conv2d->padding(luci::Padding::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_conv2d, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(ShapeRuleTest, Conv2D_sinf_dynamic)
{
  luci::CircleInput ifm;
  ifm.shape({3, 28, 28, 4});
  ifm.dim(0).unset();
  ifm.dim(1).unset();
  ifm.shape_status(luci::ShapeStatus::VALID);
  luci::CircleInput ker;
  ker.shape({2, 3, 3, 4});
  ker.shape_status(luci::ShapeStatus::VALID);
  luci::CircleInput bias;
  bias.shape({2});
  bias.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConv2D node_conv2d;
  node_conv2d.input(&ifm);
  node_conv2d.filter(&ker);
  node_conv2d.bias(&bias);
  node_conv2d.padding(luci::Padding::VALID);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;
  shape_inf_rule.infer(&node_conv2d, shape);

  ASSERT_EQ(false, shape.dim(0).known());
  ASSERT_EQ(false, shape.dim(1).known());
  ASSERT_EQ(true, shape.dim(2).known());
  ASSERT_EQ(true, shape.dim(3).known());
  ASSERT_EQ(0, shape.dim(0).value());
  ASSERT_EQ(0, shape.dim(1).value());
  ASSERT_EQ(26, shape.dim(2).value());
  ASSERT_EQ(2, shape.dim(3).value());
}

TEST(ShapeRuleTest, Conv2D_sinf_without_input_NEG)
{
  luci::CircleInput ker;
  ker.shape({2, 3, 3, 4});
  ker.shape_status(luci::ShapeStatus::VALID);
  luci::CircleInput bias;
  bias.shape({2});
  bias.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConv2D node_conv2d;
  node_conv2d.input(nullptr);
  node_conv2d.filter(&ker);
  node_conv2d.bias(&bias);
  node_conv2d.padding(luci::Padding::VALID);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;
  ASSERT_ANY_THROW(shape_inf_rule.infer(&node_conv2d, shape));
}
