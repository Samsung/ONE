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

TEST(CloneNodeTest, clone_Quantize)
{
  auto g = loco::make_graph();
  auto node_q = g->nodes()->create<luci::CircleQuantize>();

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_q, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_q = dynamic_cast<luci::CircleQuantize *>(cloned);
  ASSERT_NE(nullptr, cloned_q);
}

TEST(ShapeRuleTest, quantize_dynamic_shape)
{
  luci::CircleInput input;
  luci::CircleQuantize quantize;

  input.shape({1, 2, 3, 4});
  input.shape_status(luci::ShapeStatus::VALID);
  input.dim(2).unset();

  quantize.input(&input);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&quantize, shape));

  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_FALSE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(2, shape.dim(1).value());
  ASSERT_EQ(0, shape.dim(2).value());
  ASSERT_EQ(4, shape.dim(3).value());
}

TEST(ShapeRuleTest, quantize_shape_not_ready_NEG)
{
  luci::CircleInput input;
  luci::CircleQuantize quantize;

  input.shape_status(luci::ShapeStatus::UNDEFINED);
  quantize.input(&input);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_FALSE(shape_inf_rule.infer(&quantize, shape));
}
