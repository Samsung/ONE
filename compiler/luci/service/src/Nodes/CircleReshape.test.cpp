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

TEST(ShapeRuleTest, reshape_by_circle_const)
{
  auto g = loco::make_graph();
  auto node_reshape = g->nodes()->create<luci::CircleReshape>();
  auto tensor_input = g->nodes()->create<luci::CircleInput>();
  auto shape_input = g->nodes()->create<luci::CircleConst>();

  tensor_input->dtype(loco::DataType::S32);
  tensor_input->shape({2, 3, 4});
  tensor_input->shape_status(luci::ShapeStatus::VALID);

  shape_input->dtype(loco::DataType::S32);
  shape_input->size<loco::DataType::S32>(2);
  shape_input->at<loco::DataType::S32>(0) = -1;
  shape_input->at<loco::DataType::S32>(1) = 4;
  shape_input->shape_status(luci::ShapeStatus::VALID);

  node_reshape->tensor(tensor_input);
  node_reshape->shape(shape_input);

  loco::TensorShape output_shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(node_reshape, output_shape));

  ASSERT_EQ(2, output_shape.rank());
  ASSERT_TRUE(output_shape.dim(0).known());
  ASSERT_TRUE(output_shape.dim(1).known());
  ASSERT_EQ(6, output_shape.dim(0).value());
  ASSERT_EQ(4, output_shape.dim(1).value());
}

TEST(ShapeRuleTest, reshape_by_circle_dummy)
{
  auto g = loco::make_graph();
  auto node_reshape = g->nodes()->create<luci::CircleReshape>();
  auto tensor_input = g->nodes()->create<luci::CircleInput>();
  auto shape_input = g->nodes()->create<luci::CircleOutputDummy>();

  tensor_input->dtype(loco::DataType::S32);
  tensor_input->shape({2, 3, 4});
  tensor_input->shape_status(luci::ShapeStatus::VALID);

  shape_input->dtype(loco::DataType::S32);
  shape_input->shape_status(luci::ShapeStatus::VALID);

  node_reshape->tensor(tensor_input);
  node_reshape->shape(shape_input);
  node_reshape->newShape()->rank(2);
  node_reshape->newShape()->dim(0) = -1;
  node_reshape->newShape()->dim(1) = 4;

  loco::TensorShape output_shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(node_reshape, output_shape));

  ASSERT_EQ(2, output_shape.rank());
  ASSERT_TRUE(output_shape.dim(0).known());
  ASSERT_TRUE(output_shape.dim(1).known());
  ASSERT_EQ(6, output_shape.dim(0).value());
  ASSERT_EQ(4, output_shape.dim(1).value());
}

TEST(ShapeRuleTest, reshape_by_circle_node)
{
  auto g = loco::make_graph();
  auto node_reshape = g->nodes()->create<luci::CircleReshape>();
  auto tensor_input = g->nodes()->create<luci::CircleInput>();
  auto shape_input = g->nodes()->create<luci::CircleInput>();

  tensor_input->dtype(loco::DataType::S32);
  tensor_input->shape({2, 3, 4});
  tensor_input->shape_status(luci::ShapeStatus::VALID);

  shape_input->dtype(loco::DataType::S32);
  shape_input->shape({2});
  shape_input->shape_status(luci::ShapeStatus::VALID);

  node_reshape->tensor(tensor_input);
  node_reshape->shape(shape_input);

  loco::TensorShape output_shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(node_reshape, output_shape));

  ASSERT_EQ(2, output_shape.rank());
  ASSERT_FALSE(output_shape.dim(0).known());
  ASSERT_FALSE(output_shape.dim(1).known());
}

TEST(ShapeRuleTest, reshape_input_tensor_undefined_NEG)
{
  auto g = loco::make_graph();
  auto node_reshape = g->nodes()->create<luci::CircleReshape>();
  auto tensor_input = g->nodes()->create<luci::CircleInput>();
  auto shape_by_input = g->nodes()->create<luci::CircleConst>();

  tensor_input->dtype(loco::DataType::S32);
  tensor_input->shape({2, 3, 4});
  tensor_input->shape_status(luci::ShapeStatus::UNDEFINED);

  shape_by_input->dtype(loco::DataType::S32);
  shape_by_input->size<loco::DataType::S32>(2);
  shape_by_input->at<loco::DataType::S32>(0) = 6;
  shape_by_input->at<loco::DataType::S32>(1) = 4;
  shape_by_input->shape_status(luci::ShapeStatus::VALID);

  node_reshape->tensor(tensor_input);
  node_reshape->shape(shape_by_input);

  loco::TensorShape output_shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_FALSE(shape_inf_rule.infer(node_reshape, output_shape));
}

TEST(ShapeRuleTest, reshape_input_shape_undefined_NEG)
{
  auto g = loco::make_graph();
  auto node_reshape = g->nodes()->create<luci::CircleReshape>();
  auto tensor_input = g->nodes()->create<luci::CircleInput>();
  auto shape_by_input = g->nodes()->create<luci::CircleConst>();

  tensor_input->dtype(loco::DataType::S32);
  tensor_input->shape({2, 3, 4});
  tensor_input->shape_status(luci::ShapeStatus::VALID);

  shape_by_input->dtype(loco::DataType::S32);
  shape_by_input->size<loco::DataType::S32>(2);
  shape_by_input->at<loco::DataType::S32>(0) = 6;
  shape_by_input->at<loco::DataType::S32>(1) = 4;
  shape_by_input->shape_status(luci::ShapeStatus::UNDEFINED);

  node_reshape->tensor(tensor_input);
  node_reshape->shape(shape_by_input);

  loco::TensorShape output_shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_FALSE(shape_inf_rule.infer(node_reshape, output_shape));
}

TEST(ShapeRuleTest, reshape_no_input_shape_NEG)
{
  auto g = loco::make_graph();
  auto node_reshape = g->nodes()->create<luci::CircleReshape>();
  auto tensor_input = g->nodes()->create<luci::CircleInput>();

  tensor_input->dtype(loco::DataType::S32);
  tensor_input->shape({2, 3, 4});
  tensor_input->shape_status(luci::ShapeStatus::VALID);

  node_reshape->tensor(tensor_input);
  node_reshape->shape(nullptr);

  loco::TensorShape output_shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(node_reshape, output_shape));
}

TEST(ShapeRuleTest, reshape_too_many_unknown_NEG)
{
  auto g = loco::make_graph();
  auto node_reshape = g->nodes()->create<luci::CircleReshape>();
  auto tensor_input = g->nodes()->create<luci::CircleInput>();
  auto shape_input = g->nodes()->create<luci::CircleConst>();

  tensor_input->dtype(loco::DataType::S32);
  tensor_input->shape({2, 3, 4});
  tensor_input->shape_status(luci::ShapeStatus::VALID);

  shape_input->dtype(loco::DataType::S32);
  shape_input->size<loco::DataType::S32>(2);
  shape_input->at<loco::DataType::S32>(0) = -1;
  shape_input->at<loco::DataType::S32>(1) = -1;
  shape_input->shape_status(luci::ShapeStatus::VALID);

  node_reshape->tensor(tensor_input);
  node_reshape->shape(shape_input);

  loco::TensorShape output_shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(node_reshape, output_shape));
}
