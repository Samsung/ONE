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

#include <luci/Service/CircleShapeInference.h>

#include <gtest/gtest.h>

TEST(CloneNodeTest, clone_Concatenation)
{
  auto g = loco::make_graph();
  auto node_concat = g->nodes()->create<luci::CircleConcatenation>(3);
  node_concat->fusedActivationFunction(luci::FusedActFunc::RELU);
  node_concat->axis(7);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_concat, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_concat = dynamic_cast<luci::CircleConcatenation *>(cloned);
  ASSERT_NE(nullptr, cloned_concat);
  ASSERT_EQ(node_concat->numValues(), cloned_concat->numValues());
  ASSERT_EQ(node_concat->fusedActivationFunction(), cloned_concat->fusedActivationFunction());
  ASSERT_EQ(node_concat->axis(), cloned_concat->axis());
}

TEST(CloneNodeTest, clone_Concatenation_NEG)
{
  auto g = loco::make_graph();
  auto node_concat = g->nodes()->create<luci::CircleConcatenation>(3);
  node_concat->fusedActivationFunction(luci::FusedActFunc::UNDEFINED);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_concat, gc.get());
  ASSERT_EQ(nullptr, cloned);
}

TEST(ShapeRuleTest, concat_dynamic_shape_axis)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleConcatenation concat(2);

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 4, 3, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);
  input_2.dim(2).unset();

  concat.values(0, &input_1);
  concat.values(1, &input_2);
  concat.axis(2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&concat, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_FALSE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
  ASSERT_EQ(0, shape.dim(2).value());
  ASSERT_EQ(1, shape.dim(3).value());
}

TEST(ShapeRuleTest, concat_dynamic_shape_non_axis)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleConcatenation concat(2);

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 4, 3, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);
  input_2.dim(2).unset();

  concat.values(0, &input_1);
  concat.values(1, &input_2);
  concat.axis(1);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&concat, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(8, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(1, shape.dim(3).value());
}

TEST(ShapeRuleTest, concat_wrong_shape_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleConcatenation concat(2);

  input_1.shape({1, 4, 4, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 4, 3, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);

  concat.values(0, &input_1);
  concat.values(1, &input_2);
  concat.axis(1);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  EXPECT_ANY_THROW(shape_inf_rule.infer(&concat, shape));
}

TEST(ShapeRuleTest, concat_rank_mismatch_NEG)
{
  luci::CircleInput input_1;
  luci::CircleInput input_2;
  luci::CircleConcatenation concat(2);

  input_1.shape({1, 4, 3, 1});
  input_1.shape_status(luci::ShapeStatus::VALID);

  input_2.shape({1, 4, 1});
  input_2.shape_status(luci::ShapeStatus::VALID);
  input_2.dim(2).unset();

  concat.values(0, &input_1);
  concat.values(1, &input_2);
  concat.axis(2);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  EXPECT_ANY_THROW(shape_inf_rule.infer(&concat, shape));
}
