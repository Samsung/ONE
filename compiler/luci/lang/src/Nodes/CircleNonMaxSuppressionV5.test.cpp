/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleNonMaxSuppressionV5.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleNonMaxSuppressionV5Test, constructor)
{
  luci::CircleNonMaxSuppressionV5 nmsv5_node;

  ASSERT_EQ(luci::CircleDialect::get(), nmsv5_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::NON_MAX_SUPPRESSION_V5, nmsv5_node.opcode());

  ASSERT_EQ(nullptr, nmsv5_node.boxes());
  ASSERT_EQ(nullptr, nmsv5_node.scores());
  ASSERT_EQ(nullptr, nmsv5_node.max_output_size());
  ASSERT_EQ(nullptr, nmsv5_node.iou_threshold());
  ASSERT_EQ(nullptr, nmsv5_node.score_threshold());
  ASSERT_EQ(nullptr, nmsv5_node.soft_nms_sigma());
}

TEST(CircleNonMaxSuppressionV5Test, input_NEG)
{
  luci::CircleNonMaxSuppressionV5 nmsv5_node;
  luci::CircleNonMaxSuppressionV5 node;

  nmsv5_node.boxes(&node);
  nmsv5_node.scores(&node);
  nmsv5_node.max_output_size(&node);
  nmsv5_node.iou_threshold(&node);
  nmsv5_node.score_threshold(&node);
  nmsv5_node.soft_nms_sigma(&node);
  ASSERT_NE(nullptr, nmsv5_node.boxes());
  ASSERT_NE(nullptr, nmsv5_node.scores());
  ASSERT_NE(nullptr, nmsv5_node.max_output_size());
  ASSERT_NE(nullptr, nmsv5_node.iou_threshold());
  ASSERT_NE(nullptr, nmsv5_node.score_threshold());
  ASSERT_NE(nullptr, nmsv5_node.soft_nms_sigma());

  nmsv5_node.boxes(nullptr);
  nmsv5_node.scores(nullptr);
  nmsv5_node.max_output_size(nullptr);
  nmsv5_node.iou_threshold(nullptr);
  nmsv5_node.score_threshold(nullptr);
  nmsv5_node.soft_nms_sigma(nullptr);
  ASSERT_EQ(nullptr, nmsv5_node.boxes());
  ASSERT_EQ(nullptr, nmsv5_node.scores());
  ASSERT_EQ(nullptr, nmsv5_node.max_output_size());
  ASSERT_EQ(nullptr, nmsv5_node.iou_threshold());
  ASSERT_EQ(nullptr, nmsv5_node.score_threshold());
  ASSERT_EQ(nullptr, nmsv5_node.soft_nms_sigma());
}

TEST(CircleNonMaxSuppressionV5Test, arity_NEG)
{
  luci::CircleNonMaxSuppressionV5 nmsv5_node;

  ASSERT_NO_THROW(nmsv5_node.arg(5));
  ASSERT_THROW(nmsv5_node.arg(6), std::out_of_range);
}

TEST(CircleNonMaxSuppressionV5Test, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleNonMaxSuppressionV5 nmsv5_node;

  TestVisitor tv;
  ASSERT_THROW(nmsv5_node.accept(&tv), std::exception);
}

TEST(CircleNonMaxSuppressionV5Test, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleNonMaxSuppressionV5 nmsv5_node;

  TestVisitor tv;
  ASSERT_THROW(nmsv5_node.accept(&tv), std::exception);
}
