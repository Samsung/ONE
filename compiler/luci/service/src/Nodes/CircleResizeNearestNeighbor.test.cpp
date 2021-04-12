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

TEST(ShapeRuleTest, resize_nearest_neighbor_simple)
{
  luci::CircleInput input;
  luci::CircleConst rnn_size;
  luci::CircleResizeNearestNeighbor rnn;

  input.shape({1, 4, 4, 3});
  input.shape_status(luci::ShapeStatus::VALID);

  rnn_size.dtype(loco::DataType::S32);
  rnn_size.rank(1);
  rnn_size.dim(0).set(2);
  rnn_size.size<loco::DataType::S32>(2);
  rnn_size.at<loco::DataType::S32>(0) = 16;
  rnn_size.at<loco::DataType::S32>(1) = 16;
  rnn_size.shape_status(luci::ShapeStatus::VALID);

  rnn.input(&input);
  rnn.size(&rnn_size);

  loco::TensorShape shape;
  luci::sinf::Rule shape_inf_rule;

  ASSERT_TRUE(shape_inf_rule.infer(&rnn, shape));
  ASSERT_EQ(4, shape.rank());
  ASSERT_EQ(1, shape.dim(0).value());
  ASSERT_EQ(16, shape.dim(1).value());
  ASSERT_EQ(16, shape.dim(2).value());
  ASSERT_EQ(3, shape.dim(3).value());
}

TEST(CloneNodeTest, clone_ResizeNearestNeighbor)
{
  auto g = loco::make_graph();
  auto node_rnn = g->nodes()->create<luci::CircleResizeNearestNeighbor>();
  node_rnn->align_corners(true);

  auto gc = loco::make_graph();
  auto cloned = luci::clone_node(node_rnn, gc.get());
  ASSERT_NE(nullptr, cloned);
  ASSERT_EQ(gc.get(), cloned->graph());

  auto cloned_rnn = dynamic_cast<luci::CircleResizeNearestNeighbor *>(cloned);
  ASSERT_NE(nullptr, cloned_rnn);
  ASSERT_EQ(node_rnn->align_corners(), cloned_rnn->align_corners());
}
