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

TEST(CloneNodeTest, clone_Rsqrt)
{
  auto g = loco::make_graph();
  auto node_rsqrt = g->nodes()->create<luci::CircleRsqrt>();

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_rsqrt, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_rsqrt = dynamic_cast<luci::CircleRsqrt *>(cloned);
  ASSERT_NE(nullptr, cloned_rsqrt);
}

TEST(ShapeRuleTest, rsqrt_dynamic_shape)
{
  luci::CircleInput input;
  luci::CircleRsqrt rsqrt;

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  input.shape({1, 1, 3, 4});
  input.shape_status(luci::ShapeStatus::VALID);
  input.dim(0).unset();
  input.dim(1).unset();

  rsqrt.x(&input);

  ASSERT_TRUE(shape_inf_rule.infer(&rsqrt, shape));
  ASSERT_EQ(shape.rank(), 4);
  ASSERT_FALSE(shape.dim(0).known());
  ASSERT_FALSE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());

  ASSERT_EQ(0, shape.dim(0).value());
  ASSERT_EQ(0, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(4, shape.dim(3).value());
}

TEST(ShapeRuleTest, rsqrt_static_shape)
{
  luci::CircleInput input;
  luci::CircleRsqrt rsqrt;

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  input.shape({3, 2, 3, 4});
  input.shape_status(luci::ShapeStatus::VALID);

  rsqrt.x(&input);

  ASSERT_TRUE(shape_inf_rule.infer(&rsqrt, shape));
  ASSERT_EQ(shape.rank(), 4);
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());

  ASSERT_EQ(3, shape.dim(0).value());
  ASSERT_EQ(2, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(4, shape.dim(3).value());
}

TEST(ShapeRuleTest, rsqrt_nullptr_input_NEG)
{
  luci::CircleRsqrt rsqrt;

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  rsqrt.x(nullptr);
  ASSERT_ANY_THROW(shape_inf_rule.infer(&rsqrt, shape));
}

TEST(ShapeRuleTest, rsqrt_shape_not_ready_NEG)
{
  luci::CircleInput input;
  luci::CircleRsqrt rsqrt;

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  input.shape_status(luci::ShapeStatus::UNDEFINED);

  rsqrt.x(&input);
  ASSERT_FALSE(shape_inf_rule.infer(&rsqrt, shape));
}
