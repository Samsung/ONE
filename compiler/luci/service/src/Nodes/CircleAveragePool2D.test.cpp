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
