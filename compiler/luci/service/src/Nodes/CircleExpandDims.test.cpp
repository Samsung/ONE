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

#include <luci/IR/CircleNodes.h>
#include <luci/Service/CircleShapeInference.h>

#include <loco/IR/TensorShape.h>

#include <gtest/gtest.h>

TEST(ShapeRuleTest, simple_expand_dims)
{
  luci::CircleInput input;
  luci::CircleConst axis;
  luci::CircleExpandDims expand_dims;

  input.shape({4, 3});
  input.shape_status(luci::ShapeStatus::VALID);

  axis.dtype(loco::DataType::S32);
  axis.rank(0);
  axis.size<loco::DataType::S32>(1);
  axis.at<loco::DataType::S32>(0) = 1;
  axis.shape_status(luci::ShapeStatus::VALID);

  expand_dims.input(&input);
  expand_dims.axis(&axis);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&expand_dims, shape));
  ASSERT_EQ(3, shape.rank());
  ASSERT_EQ(4, shape.dim(0).value());
  ASSERT_EQ(1, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
}

TEST(CloneNodeTest, clone_ExpandDims)
{
  auto g = loco::make_graph();
  auto node_ed = g->nodes()->create<luci::CircleExpandDims>();

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_ed, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_ed = dynamic_cast<luci::CircleExpandDims *>(cloned);
  ASSERT_NE(nullptr, cloned_ed);
}
