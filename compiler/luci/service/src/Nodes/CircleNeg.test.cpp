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

TEST(CloneNodeTest, clone_Neg)
{
  auto g = loco::make_graph();
  auto node_neg = g->nodes()->create<luci::CircleNeg>();

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_neg, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_neg = dynamic_cast<luci::CircleNeg *>(cloned);
  ASSERT_NE(nullptr, cloned_neg);
}

TEST(ShapeRuleTest, Neg_dynamic_shape)
{
  luci::CircleInput input;
  luci::CircleNeg neg;

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  input.shape({1, 1, 3, 4});
  input.shape_status(luci::ShapeStatus::VALID);
  input.dim(1).unset();

  neg.x(&input);

  ASSERT_TRUE(shape_inf_rule.infer(&neg, shape));
  ASSERT_EQ(shape.rank(), 4);
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_FALSE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());

  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(0, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
  ASSERT_EQ(4, shape.dim(3).value());
}

TEST(ShapeRuleTest, Neg_nullptr_input_NEG)
{
  luci::CircleNeg neg;

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  neg.x(nullptr);
  ASSERT_ANY_THROW(shape_inf_rule.infer(&neg, shape));
}
