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

/**
 * @note Function to test: Shape inference of two different input shapes
 *
 *       Rank expansion to higher input side
 *          x(2,1,5) + y(3,5) --> x(2,1,5) + y(1,3,5)
 *       Do output shape inference like numpy
 *          x(2,1,5) + y(1,3,5) --> output(2,3,5)
 *       For each axis, dim value should be same OR one of them should be 1
 */
TEST(ShapeRuleTest, different_input_shapes_add)
{
  luci::CircleInput input1;
  luci::CircleInput input2;
  luci::CircleAdd add;

  input1.shape({2, 1, 5});
  input1.shape_status(luci::ShapeStatus::VALID);
  input2.shape({3, 5});
  input2.shape_status(luci::ShapeStatus::VALID);

  add.x(&input1);
  add.y(&input2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&add, shape));
  ASSERT_EQ(3, shape.rank());
  ASSERT_EQ(2, shape.dim(0).value());
  ASSERT_EQ(3, shape.dim(1).value());
  ASSERT_EQ(5, shape.dim(2).value());
}

TEST(ShapeRuleTest, add_dynamic_shape_non_1)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleAdd add;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 1, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);
  input_2.dim(0).unset();

  add.x(&input_1);
  add.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&add, shape));
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

TEST(ShapeRuleTest, add_dynamic_shape_1)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleAdd add;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 1, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);
  input_2.dim(2).unset();

  add.x(&input_1);
  add.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&add, shape));
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

TEST(ShapeRuleTest, add_dynamic_shape_both)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleAdd add;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);
  input_1.dim(3).unset();

  input_2.shape({1, 1, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);
  input_2.dim(2).unset();

  add.x(&input_1);
  add.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&add, shape));
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

TEST(ShapeRuleTest, add_scalar)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleAdd add;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({});
  input_2.shape_status(luci::ShapeStatus::VALID);

  add.x(&input_1);
  add.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&add, shape));
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

TEST(ShapeRuleTest, add_not_broadcastable_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleAdd add;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 2, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);

  add.x(&input_1);
  add.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&add, shape));
}

TEST(ShapeRuleTest, add_not_broadcastable_2_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleAdd add;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({2, 1, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);

  add.x(&input_1);
  add.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&add, shape));
}

TEST(ShapeRuleTest, add_not_broadcastable_3_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleAdd add;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({2, 3, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);

  add.x(&input_1);
  add.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&add, shape));
}

TEST(ShapeRuleTest, add_not_broadcastable_4_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleAdd add;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({2, 3, 2});
  input_2.shape_status(luci::ShapeStatus::VALID);

  add.x(&input_1);
  add.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&add, shape));
}

TEST(ShapeRuleTest, add_not_broadcastable_5_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleAdd add;

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({3, 2, 3, 2});
  input_2.shape_status(luci::ShapeStatus::VALID);

  add.x(&input_1);
  add.y(&input_2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&add, shape));
}

TEST(CloneNodeTest, clone_Add)
{
  auto g = loco::make_graph();
  auto node_add = g->nodes()->create<luci::CircleAdd>();
  node_add->fusedActivationFunction(luci::FusedActFunc::RELU);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_add, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_add = dynamic_cast<luci::CircleAdd *>(cloned);
  ASSERT_NE(nullptr, cloned_add);
  ASSERT_EQ(node_add->fusedActivationFunction(), cloned_add->fusedActivationFunction());
}

TEST(CloneNodeTest, clone_Add_NEG)
{
  auto g = loco::make_graph();
  auto node_add = g->nodes()->create<luci::CircleAdd>();
  node_add->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_add, gc.get());
  ASSERT_EQ(nullptr, cloned);
}
