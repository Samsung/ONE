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

TEST(ShapeRuleTest, transpose_simple)
{
  luci::CircleInput input;
  luci::CircleConst perm;
  luci::CircleTranspose transpose;

  input.shape({3, 8, 1});
  input.shape_status(luci::ShapeStatus::VALID);

  perm.dtype(loco::DataType::S32);
  perm.rank(1);
  perm.dim(0).set(3);
  perm.size<loco::DataType::S32>(3);
  perm.at<loco::DataType::S32>(0) = 1;
  perm.at<loco::DataType::S32>(1) = 2;
  perm.at<loco::DataType::S32>(2) = 0;
  perm.shape_status(luci::ShapeStatus::VALID);

  transpose.a(&input);
  transpose.perm(&perm);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&transpose, shape));
  ASSERT_EQ(3, shape.rank());
  ASSERT_EQ(8, shape.dim(0).value());
  ASSERT_EQ(1, shape.dim(1).value());
  ASSERT_EQ(3, shape.dim(2).value());
}

TEST(CloneNodeTest, clone_Transpose)
{
  auto g = loco::make_graph();
  auto node_tr = g->nodes()->create<luci::CircleTranspose>();

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_tr, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_tr = dynamic_cast<luci::CircleTranspose *>(cloned);
  ASSERT_NE(nullptr, cloned_tr);
}
