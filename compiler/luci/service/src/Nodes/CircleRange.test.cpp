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

TEST(CloneNodeTest, clone_Range)
{
  auto g = loco::make_graph();
  auto node_range = g->nodes()->create<luci::CircleRange>();

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_range, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_range = dynamic_cast<luci::CircleRange *>(cloned);
  ASSERT_NE(nullptr, cloned_range);
}

TEST(ShapeRuleTest, range_const_param)
{
  luci::CircleConst start, limit, delta;
  luci::CircleRange range;

  start.dtype(loco::DataType::S32);
  start.size<loco::DataType::S32>(1);
  start.at<loco::DataType::S32>(0) = 0;
  start.shape_status(luci::ShapeStatus::VALID);

  limit.dtype(loco::DataType::S32);
  limit.size<loco::DataType::S32>(1);
  limit.at<loco::DataType::S32>(0) = 10;
  limit.shape_status(luci::ShapeStatus::VALID);

  delta.dtype(loco::DataType::S32);
  delta.size<loco::DataType::S32>(1);
  delta.at<loco::DataType::S32>(0) = 2;
  delta.shape_status(luci::ShapeStatus::VALID);

  range.start(&start);
  range.limit(&limit);
  range.delta(&delta);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&range, shape));
  ASSERT_EQ(1, shape.rank());
  ASSERT_TRUE(shape.dim(0).known());
  ASSERT_EQ(5, shape.dim(0).value());
}

TEST(ShapeRuleTest, range_zero_delta_NEG)
{
  luci::CircleConst start, limit, delta;
  luci::CircleRange range;

  start.dtype(loco::DataType::S32);
  start.size<loco::DataType::S32>(1);
  start.at<loco::DataType::S32>(0) = 0;
  start.shape_status(luci::ShapeStatus::VALID);

  limit.dtype(loco::DataType::S32);
  limit.size<loco::DataType::S32>(1);
  limit.at<loco::DataType::S32>(0) = 10;
  limit.shape_status(luci::ShapeStatus::VALID);

  delta.dtype(loco::DataType::S32);
  delta.size<loco::DataType::S32>(1);
  delta.at<loco::DataType::S32>(0) = 0;
  delta.shape_status(luci::ShapeStatus::VALID);

  range.start(&start);
  range.limit(&limit);
  range.delta(&delta);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&range, shape));
}

TEST(ShapeRuleTest, range_non_const_param)
{
  luci::CircleInput start, limit, delta;
  luci::CircleRange range;

  start.dtype(loco::DataType::S32);
  start.shape({1});
  start.shape_status(luci::ShapeStatus::VALID);

  limit.dtype(loco::DataType::S32);
  limit.shape({1});
  limit.shape_status(luci::ShapeStatus::VALID);

  delta.dtype(loco::DataType::S32);
  delta.shape({1});
  delta.shape_status(luci::ShapeStatus::VALID);

  range.start(&start);
  range.limit(&limit);
  range.delta(&delta);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&range, shape));
  ASSERT_EQ(1, shape.rank());
  ASSERT_FALSE(shape.dim(0).known());
  ASSERT_EQ(0, shape.dim(0).value());
}

TEST(ShapeRuleTest, range_nullptr_start_NEG)
{
  luci::CircleInput limit, delta;
  luci::CircleRange range;

  limit.dtype(loco::DataType::S32);
  limit.shape({1});
  limit.shape_status(luci::ShapeStatus::VALID);

  delta.dtype(loco::DataType::S32);
  delta.shape({1});
  delta.shape_status(luci::ShapeStatus::VALID);

  range.start(nullptr);
  range.limit(&limit);
  range.delta(&delta);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_ANY_THROW(shape_inf_rule.infer(&range, shape));
}
