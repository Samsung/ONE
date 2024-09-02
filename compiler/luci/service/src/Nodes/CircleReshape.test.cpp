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

TEST(CloneNodeTest, clone_Reshape)
{
  auto g = loco::make_graph();
  auto node_reshape = g->nodes()->create<luci::CircleReshape>();
  node_reshape->newShape()->rank(2);
  node_reshape->newShape()->dim(0) = 3;
  node_reshape->newShape()->dim(1) = 4;

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_reshape, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_reshape = dynamic_cast<luci::CircleReshape *>(cloned);
  ASSERT_NE(nullptr, cloned_reshape);
  ASSERT_EQ(node_reshape->newShape()->rank(), cloned_reshape->newShape()->rank());
  ASSERT_EQ(node_reshape->newShape()->dim(0), cloned_reshape->newShape()->dim(0));
  ASSERT_EQ(node_reshape->newShape()->dim(1), cloned_reshape->newShape()->dim(1));
}

TEST(ShapeRuleTest, reshape_both_known_dim)
{
  luci::CircleInput input;
  input.shape({2, 3, 4});
  input.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConst new_shape;
  new_shape.dtype(loco::DataType::S32);
  new_shape.size<loco::DataType::S32>(2);
  new_shape.at<loco::DataType::S32>(0) = 6;
  new_shape.at<loco::DataType::S32>(1) = 4;
  new_shape.shape_status(luci::ShapeStatus::VALID);

  luci::CircleReshape reshape;
  reshape.tensor(&input);
  reshape.shape(&new_shape);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&reshape, shape));

  ASSERT_EQ(3, input.rank());
  ASSERT_TRUE(input.dim(0).known());
  ASSERT_TRUE(input.dim(1).known());
  ASSERT_TRUE(input.dim(2).known());
  ASSERT_EQ(2, input.dim(0).value());
  ASSERT_EQ(3, input.dim(1).value());
  ASSERT_EQ(4, input.dim(2).value());
  ASSERT_EQ(2, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_EQ(6, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
}

TEST(ShapeRuleTest, reshape_input_unknown_dim)
{
  luci::CircleInput input;
  input.shape({2, 3, 4});
  input.dim(0).unset();
  input.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConst new_shape;
  new_shape.dtype(loco::DataType::S32);
  new_shape.size<loco::DataType::S32>(2);
  new_shape.at<loco::DataType::S32>(0) = 6;
  new_shape.at<loco::DataType::S32>(1) = 4;
  new_shape.shape_status(luci::ShapeStatus::VALID);

  luci::CircleReshape reshape;
  reshape.tensor(&input);
  reshape.shape(&new_shape);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&reshape, shape));

  ASSERT_EQ(3, input.rank());
  ASSERT_FALSE(input.dim(0).known());
  ASSERT_TRUE(input.dim(1).known());
  ASSERT_TRUE(input.dim(2).known());
  ASSERT_EQ(0, input.dim(0).value());
  ASSERT_EQ(3, input.dim(1).value());
  ASSERT_EQ(4, input.dim(2).value());
  ASSERT_EQ(2, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_EQ(6, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
}

TEST(ShapeRuleTest, reshape_output_unknown_dim)
{
  luci::CircleInput input;
  input.shape({2, 3, 4});
  input.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConst new_shape;
  new_shape.dtype(loco::DataType::S32);
  new_shape.size<loco::DataType::S32>(2);
  new_shape.at<loco::DataType::S32>(0) = -1;
  new_shape.at<loco::DataType::S32>(1) = 4;
  new_shape.shape_status(luci::ShapeStatus::VALID);

  luci::CircleReshape reshape;
  reshape.tensor(&input);
  reshape.shape(&new_shape);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&reshape, shape));

  ASSERT_EQ(3, input.rank());
  ASSERT_TRUE(input.dim(0).known());
  ASSERT_TRUE(input.dim(1).known());
  ASSERT_TRUE(input.dim(2).known());
  ASSERT_EQ(2, input.dim(0).value());
  ASSERT_EQ(3, input.dim(1).value());
  ASSERT_EQ(4, input.dim(2).value());
  ASSERT_EQ(2, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_EQ(6, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
}

TEST(ShapeRuleTest, reshape_both_unknown_dim)
{
  luci::CircleInput input;
  input.shape({2, 3, 4});
  input.dim(0).unset();
  input.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConst new_shape;
  new_shape.dtype(loco::DataType::S32);
  new_shape.size<loco::DataType::S32>(2);
  new_shape.at<loco::DataType::S32>(0) = -1;
  new_shape.at<loco::DataType::S32>(1) = 4;
  new_shape.shape_status(luci::ShapeStatus::VALID);

  luci::CircleReshape reshape;
  reshape.tensor(&input);
  reshape.shape(&new_shape);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&reshape, shape));

  ASSERT_EQ(3, input.rank());
  ASSERT_FALSE(input.dim(0).known());
  ASSERT_TRUE(input.dim(1).known());
  ASSERT_TRUE(input.dim(2).known());
  ASSERT_EQ(0, input.dim(0).value());
  ASSERT_EQ(3, input.dim(1).value());
  ASSERT_EQ(4, input.dim(2).value());
  ASSERT_EQ(2, shape.rank());
  ASSERT_FALSE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_EQ(0, shape.dim(0).value());
  ASSERT_EQ(4, shape.dim(1).value());
}

TEST(ShapeRuleTest, reshape_output_multiple_unknown_dim_1_NEG)
{
  luci::CircleInput input;
  input.shape({2, 3, 4});
  input.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConst new_shape;
  new_shape.dtype(loco::DataType::S32);
  new_shape.size<loco::DataType::S32>(2);
  new_shape.at<loco::DataType::S32>(0) = -1;
  new_shape.at<loco::DataType::S32>(1) = -1;
  new_shape.shape_status(luci::ShapeStatus::VALID);

  luci::CircleReshape reshape;
  reshape.tensor(&input);
  reshape.shape(&new_shape);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&reshape, shape));
}

TEST(ShapeRuleTest, reshape_output_multiple_unknown_dim_2_NEG)
{
  luci::CircleInput input;
  input.shape({2, 3, 4});
  input.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConst new_shape;
  new_shape.dtype(loco::DataType::S32);
  new_shape.size<loco::DataType::S32>(2);
  new_shape.at<loco::DataType::S32>(0) = -2;
  new_shape.at<loco::DataType::S32>(1) = -2;
  new_shape.shape_status(luci::ShapeStatus::VALID);

  luci::CircleReshape reshape;
  reshape.tensor(&input);
  reshape.shape(&new_shape);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&reshape, shape));
}

TEST(ShapeRuleTest, reshape_output_multiple_unknown_dim_3_NEG)
{
  luci::CircleInput input;
  input.shape({2, 3, 4});
  input.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConst new_shape;
  new_shape.dtype(loco::DataType::S32);
  new_shape.size<loco::DataType::S32>(2);
  new_shape.at<loco::DataType::S32>(0) = INT32_MIN;
  new_shape.at<loco::DataType::S32>(1) = INT32_MIN;
  new_shape.shape_status(luci::ShapeStatus::VALID);

  luci::CircleReshape reshape;
  reshape.tensor(&input);
  reshape.shape(&new_shape);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&reshape, shape));
}

TEST(ShapeRuleTest, reshape_output_unknown_dim_4_NEG)
{
  luci::CircleInput input;
  input.shape({2, 3, 4});
  input.shape_status(luci::ShapeStatus::VALID);

  luci::CircleConst new_shape;
  new_shape.dtype(loco::DataType::S32);
  new_shape.size<loco::DataType::S32>(2);
  new_shape.at<loco::DataType::S32>(0) = INT32_MIN + 1;
  new_shape.at<loco::DataType::S32>(1) = INT32_MIN + 1;
  new_shape.shape_status(luci::ShapeStatus::VALID);

  luci::CircleReshape reshape;
  reshape.tensor(&input);
  reshape.shape(&new_shape);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&reshape, shape));
}
