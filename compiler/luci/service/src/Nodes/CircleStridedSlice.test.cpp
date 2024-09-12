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

TEST(CloneNodeTest, clone_StridedSlice)
{
  auto g = loco::make_graph();
  auto node_ss = g->nodes()->create<luci::CircleStridedSlice>();
  node_ss->begin_mask(1);
  node_ss->end_mask(2);
  node_ss->ellipsis_mask(3);
  node_ss->new_axis_mask(4);
  node_ss->shrink_axis_mask(5);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_ss, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_ss = dynamic_cast<luci::CircleStridedSlice *>(cloned);
  ASSERT_NE(nullptr, cloned_ss);
  ASSERT_EQ(node_ss->begin_mask(), cloned_ss->begin_mask());
  ASSERT_EQ(node_ss->end_mask(), cloned_ss->end_mask());
  ASSERT_EQ(node_ss->ellipsis_mask(), cloned_ss->ellipsis_mask());
  ASSERT_EQ(node_ss->new_axis_mask(), cloned_ss->new_axis_mask());
  ASSERT_EQ(node_ss->shrink_axis_mask(), cloned_ss->shrink_axis_mask());
}

TEST(ShapeRuleTest, strided_slice_static_shape)
{
  luci::CircleInput input;
  luci::CircleConst begin;
  luci::CircleConst end;
  luci::CircleConst strides;

  luci::CircleStridedSlice strided_slice;

  input.shape({5, 6, 7, 8});
  input.shape_status(luci::ShapeStatus::VALID);

  begin.dtype(loco::DataType::S32);
  begin.shape({4});
  begin.size<loco::DataType::S32>(4);
  begin.at<loco::DataType::S32>(0) = 0;
  begin.at<loco::DataType::S32>(1) = 0;
  begin.at<loco::DataType::S32>(2) = 0;
  begin.at<loco::DataType::S32>(3) = 0;
  begin.shape_status(luci::ShapeStatus::VALID);

  end.dtype(loco::DataType::S32);
  end.shape({4});
  end.size<loco::DataType::S32>(4);
  end.at<loco::DataType::S32>(0) = 1;
  end.at<loco::DataType::S32>(1) = 3;
  end.at<loco::DataType::S32>(2) = 5;
  end.at<loco::DataType::S32>(3) = 7;
  end.shape_status(luci::ShapeStatus::VALID);

  strides.dtype(loco::DataType::S32);
  strides.shape({4});
  strides.size<loco::DataType::S32>(4);
  strides.at<loco::DataType::S32>(0) = 1;
  strides.at<loco::DataType::S32>(1) = 1;
  strides.at<loco::DataType::S32>(2) = 1;
  strides.at<loco::DataType::S32>(3) = 1;
  strides.shape_status(luci::ShapeStatus::VALID);

  strided_slice.input(&input);
  strided_slice.begin(&begin);
  strided_slice.end(&end);
  strided_slice.strides(&strides);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&strided_slice, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_TRUE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(3, shape.dim(1).value());
  ASSERT_EQ(5, shape.dim(2).value());
  ASSERT_EQ(7, shape.dim(3).value());
}

TEST(ShapeRuleTest, strided_slice_non_const_begin_end)
{
  luci::CircleInput input;
  luci::CircleInput begin;
  luci::CircleInput end;
  luci::CircleConst strides;

  luci::CircleStridedSlice strided_slice;

  input.shape({5, 6, 7, 8});
  input.shape_status(luci::ShapeStatus::VALID);

  begin.dtype(loco::DataType::S32);
  begin.shape({4});
  begin.shape_status(luci::ShapeStatus::VALID);

  end.dtype(loco::DataType::S32);
  end.shape({4});
  end.shape_status(luci::ShapeStatus::VALID);
  end.dim(0).unset();

  strides.dtype(loco::DataType::S32);
  strides.shape({4});
  strides.size<loco::DataType::S32>(4);
  strides.at<loco::DataType::S32>(0) = 1;
  strides.at<loco::DataType::S32>(1) = 1;
  strides.at<loco::DataType::S32>(2) = 1;
  strides.at<loco::DataType::S32>(3) = 1;
  strides.shape_status(luci::ShapeStatus::VALID);

  strided_slice.input(&input);
  strided_slice.begin(&begin);
  strided_slice.end(&end);
  strided_slice.strides(&strides);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&strided_slice, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_FALSE(shape.dim(0).known());
  ASSERT_FALSE(shape.dim(1).known());
  ASSERT_FALSE(shape.dim(2).known());
  ASSERT_FALSE(shape.dim(3).known());
  ASSERT_EQ(0, shape.dim(0).value());
  ASSERT_EQ(0, shape.dim(1).value());
  ASSERT_EQ(0, shape.dim(2).value());
  ASSERT_EQ(0, shape.dim(3).value());
}

TEST(ShapeRuleTest, strided_slice_dynamic_input)
{
  luci::CircleInput input;
  luci::CircleConst begin;
  luci::CircleConst end;
  luci::CircleConst strides;

  luci::CircleStridedSlice strided_slice;

  input.shape({5, 6, 7, 8});
  input.shape_status(luci::ShapeStatus::VALID);
  input.dim(2).unset();

  begin.dtype(loco::DataType::S32);
  begin.shape({4});
  begin.size<loco::DataType::S32>(4);
  begin.at<loco::DataType::S32>(0) = 0;
  begin.at<loco::DataType::S32>(1) = 0;
  begin.at<loco::DataType::S32>(2) = 0;
  begin.at<loco::DataType::S32>(3) = 0;
  begin.shape_status(luci::ShapeStatus::VALID);

  end.dtype(loco::DataType::S32);
  end.shape({4});
  end.size<loco::DataType::S32>(4);
  end.at<loco::DataType::S32>(0) = 1;
  end.at<loco::DataType::S32>(1) = 3;
  end.at<loco::DataType::S32>(2) = 5;
  end.at<loco::DataType::S32>(3) = 7;
  end.shape_status(luci::ShapeStatus::VALID);

  strides.dtype(loco::DataType::S32);
  strides.shape({4});
  strides.size<loco::DataType::S32>(4);
  strides.at<loco::DataType::S32>(0) = 1;
  strides.at<loco::DataType::S32>(1) = 1;
  strides.at<loco::DataType::S32>(2) = 1;
  strides.at<loco::DataType::S32>(3) = 1;
  strides.shape_status(luci::ShapeStatus::VALID);

  strided_slice.input(&input);
  strided_slice.begin(&begin);
  strided_slice.end(&end);
  strided_slice.strides(&strides);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&strided_slice, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_TRUE(shape.dim(1).known());
  ASSERT_FALSE(shape.dim(2).known());
  ASSERT_TRUE(shape.dim(3).known());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(3, shape.dim(1).value());
  ASSERT_EQ(0, shape.dim(2).value());
  ASSERT_EQ(7, shape.dim(3).value());
}

TEST(ShapeRuleTest, strided_slice_nullptr_input_NEG)
{
  luci::CircleConst begin;
  luci::CircleConst end;
  luci::CircleConst strides;

  luci::CircleStridedSlice strided_slice;

  begin.dtype(loco::DataType::S32);
  begin.shape({4});
  begin.size<loco::DataType::S32>(4);
  begin.at<loco::DataType::S32>(0) = 0;
  begin.at<loco::DataType::S32>(1) = 0;
  begin.at<loco::DataType::S32>(2) = 0;
  begin.at<loco::DataType::S32>(3) = 0;
  begin.shape_status(luci::ShapeStatus::VALID);

  end.dtype(loco::DataType::S32);
  end.shape({4});
  end.size<loco::DataType::S32>(4);
  end.at<loco::DataType::S32>(0) = 1;
  end.at<loco::DataType::S32>(1) = 3;
  end.at<loco::DataType::S32>(2) = 5;
  end.at<loco::DataType::S32>(3) = 0;
  end.shape_status(luci::ShapeStatus::VALID);

  strides.dtype(loco::DataType::S32);
  strides.shape({4});
  strides.size<loco::DataType::S32>(4);
  strides.at<loco::DataType::S32>(0) = 1;
  strides.at<loco::DataType::S32>(1) = 1;
  strides.at<loco::DataType::S32>(2) = 1;
  strides.at<loco::DataType::S32>(3) = 1;
  strides.shape_status(luci::ShapeStatus::VALID);

  strided_slice.input(nullptr);
  strided_slice.begin(&begin);
  strided_slice.end(&end);
  strided_slice.strides(&strides);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&strided_slice, shape));
}

TEST(ShapeRuleTest, strided_slice_nullptr_begin_NEG)
{
  luci::CircleInput input;
  luci::CircleConst end;
  luci::CircleConst strides;

  luci::CircleStridedSlice strided_slice;

  input.shape({5, 6, 7, 8});
  input.shape_status(luci::ShapeStatus::VALID);

  end.dtype(loco::DataType::S32);
  end.shape({4});
  end.size<loco::DataType::S32>(4);
  end.at<loco::DataType::S32>(0) = 1;
  end.at<loco::DataType::S32>(1) = 3;
  end.at<loco::DataType::S32>(2) = 5;
  end.at<loco::DataType::S32>(3) = 0;
  end.shape_status(luci::ShapeStatus::VALID);

  strides.dtype(loco::DataType::S32);
  strides.shape({4});
  strides.size<loco::DataType::S32>(4);
  strides.at<loco::DataType::S32>(0) = 1;
  strides.at<loco::DataType::S32>(1) = 1;
  strides.at<loco::DataType::S32>(2) = 1;
  strides.at<loco::DataType::S32>(3) = 1;
  strides.shape_status(luci::ShapeStatus::VALID);

  strided_slice.input(&input);
  strided_slice.begin(nullptr);
  strided_slice.end(&end);
  strided_slice.strides(&strides);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&strided_slice, shape));
}

TEST(ShapeRuleTest, strided_slice_nullptr_end_NEG)
{
  luci::CircleInput input;
  luci::CircleConst begin;
  luci::CircleConst strides;

  luci::CircleStridedSlice strided_slice;

  input.shape({5, 6, 7, 8});
  input.shape_status(luci::ShapeStatus::VALID);

  begin.dtype(loco::DataType::S32);
  begin.shape({4});
  begin.size<loco::DataType::S32>(4);
  begin.at<loco::DataType::S32>(0) = 0;
  begin.at<loco::DataType::S32>(1) = 0;
  begin.at<loco::DataType::S32>(2) = 0;
  begin.at<loco::DataType::S32>(3) = 0;
  begin.shape_status(luci::ShapeStatus::VALID);

  strides.dtype(loco::DataType::S32);
  strides.shape({4});
  strides.size<loco::DataType::S32>(4);
  strides.at<loco::DataType::S32>(0) = 1;
  strides.at<loco::DataType::S32>(1) = 1;
  strides.at<loco::DataType::S32>(2) = 1;
  strides.at<loco::DataType::S32>(3) = 1;
  strides.shape_status(luci::ShapeStatus::VALID);

  strided_slice.input(&input);
  strided_slice.begin(&begin);
  strided_slice.end(nullptr);
  strided_slice.strides(&strides);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&strided_slice, shape));
}

TEST(ShapeRuleTest, strided_slice_nullptr_strides_NEG)
{
  luci::CircleInput input;
  luci::CircleConst begin;
  luci::CircleConst end;

  luci::CircleStridedSlice strided_slice;

  input.shape({5, 6, 7, 8});
  input.shape_status(luci::ShapeStatus::VALID);

  begin.dtype(loco::DataType::S32);
  begin.shape({4});
  begin.size<loco::DataType::S32>(4);
  begin.at<loco::DataType::S32>(0) = 0;
  begin.at<loco::DataType::S32>(1) = 0;
  begin.at<loco::DataType::S32>(2) = 0;
  begin.at<loco::DataType::S32>(3) = 0;
  begin.shape_status(luci::ShapeStatus::VALID);

  end.dtype(loco::DataType::S32);
  end.shape({4});
  end.size<loco::DataType::S32>(4);
  end.at<loco::DataType::S32>(0) = 1;
  end.at<loco::DataType::S32>(1) = 3;
  end.at<loco::DataType::S32>(2) = 5;
  end.at<loco::DataType::S32>(3) = 0;
  end.shape_status(luci::ShapeStatus::VALID);

  strided_slice.input(&input);
  strided_slice.begin(&begin);
  strided_slice.end(&end);
  strided_slice.strides(nullptr);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&strided_slice, shape));
}
