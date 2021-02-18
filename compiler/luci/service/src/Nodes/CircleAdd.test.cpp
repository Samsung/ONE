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
