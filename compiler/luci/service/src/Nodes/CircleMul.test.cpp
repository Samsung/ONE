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

TEST(CloneNodeTest, clone_Mul)
{
  auto g = loco::make_graph();
  auto node_mul = g->nodes()->create<luci::CircleMul>();
  node_mul->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_mul, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_mul = dynamic_cast<luci::CircleMul *>(cloned);
  ASSERT_NE(nullptr, cloned_mul);
  ASSERT_EQ(node_mul->fusedActivationFunction(), cloned_mul->fusedActivationFunction());
}

TEST(CloneNodeTest, clone_Mul_NEG)
{
  auto g = loco::make_graph();
  auto node_mul = g->nodes()->create<luci::CircleMul>();
  node_mul->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_mul, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(ShapeRuleTest, mul_known_dim)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleMul mul;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 1, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);

  mul.x(&input_1);
  mul.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&mul, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(1, shape.dim(3).value());
}

TEST(ShapeRuleTest, mul_dynamic_shape_non_1)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleMul mul;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 1, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);
  input_2.dim(0).unset();

  mul.x(&input_1);
  mul.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&mul, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(1, shape.dim(3).value());
}

TEST(ShapeRuleTest, mul_dynamic_shape_1)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleMul mul;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 1, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);
  input_2.dim(2).unset();

  mul.x(&input_1);
  mul.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&mul, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_FALSE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(0, shape.dim(3).value());
}

TEST(ShapeRuleTest, mul_dynamic_shape_both)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleMul mul;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);
  input_1.dim(3).unset();

  input_2.shape({1, 1, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);
  input_2.dim(2).unset();

  mul.x(&input_1);
  mul.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&mul, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_FALSE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(0, shape.dim(3).value());
}

TEST(ShapeRuleTest, mul_scalar)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleMul mul;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({});
  input_2.shape_status(luci::ShapeStatus::VALID);

  mul.x(&input_1);
  mul.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&mul, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(1, shape.dim(3).value());
}

TEST(ShapeRuleTest, mul_not_broadcastable_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleMul mul;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 2, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);

  mul.x(&input_1);
  mul.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&mul, shape));
}

TEST(ShapeRuleTest, mul_not_broadcastable_2_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleMul mul;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({2, 1, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);

  mul.x(&input_1);
  mul.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&mul, shape));
}

TEST(ShapeRuleTest, mul_not_broadcastable_3_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleMul mul;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({2, 3, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);

  mul.x(&input_1);
  mul.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&mul, shape));
}

TEST(ShapeRuleTest, mul_not_broadcastable_4_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleMul mul;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({2, 3, 2});
  input_2.shape_status(luci::ShapeStatus::VALID);

  mul.x(&input_1);
  mul.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&mul, shape));
}

TEST(ShapeRuleTest, mul_not_broadcastable_5_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleMul mul;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({3, 2, 3, 2});
  input_2.shape_status(luci::ShapeStatus::VALID);

  mul.x(&input_1);
  mul.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&mul, shape));
}
