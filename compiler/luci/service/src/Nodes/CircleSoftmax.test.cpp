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

TEST(CloneNodeTest, clone_Softmax)
{
  auto g = loco::make_graph();
  auto node_sm = g->nodes()->create<luci::CircleSoftmax>();
  node_sm->beta(2.3f);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_sm, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_sm = dynamic_cast<luci::CircleSoftmax *>(cloned);
  ASSERT_NE(nullptr, cloned_sm);
  ASSERT_EQ(node_sm->beta(), cloned_sm->beta());
}

TEST(ShapeRuleTest, softmax_static_shape)
{
  luci::CircleInput input;
  luci::CircleSoftmax softmax;

  input.shape({1, 4, 3, 8});
  input.shape_status(luci::ShapeStatus::VALID);

  softmax.logits(&input);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&softmax, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(8, shape.dim(3).value());
}

TEST(ShapeRuleTest, softmax_dynamic_shape)
{
  luci::CircleInput input;
  luci::CircleSoftmax softmax;

  input.shape({1, 4, 3, 8});
  input.shape_status(luci::ShapeStatus::VALID);
  input.dim(1).unset();

  softmax.logits(&input);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&softmax, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_FALSE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(0, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(8, shape.dim(3).value());
}

TEST(ShapeRuleTest, softmax_wrong_input_NEG)
{
  luci::CircleSoftmax softmax;

  softmax.logits(nullptr);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&softmax, shape));
}

TEST(ShapeRuleTest, softmax_wrong_input_2_NEG)
{
  luci::CircleInput *input = nullptr;
  luci::CircleSoftmax softmax;

  softmax.logits(input);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&softmax, shape));
}
