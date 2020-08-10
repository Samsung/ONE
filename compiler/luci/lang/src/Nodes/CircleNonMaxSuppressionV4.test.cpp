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

#include "luci/IR/Nodes/CircleNonMaxSuppressionV4.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleNonMaxSuppressionV4Test, constructor)
{
  luci::CircleNonMaxSuppressionV4 nmsv4_node;

  ASSERT_EQ(luci::CircleDialect::get(), nmsv4_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::NON_MAX_SUPPRESSION_V4, nmsv4_node.opcode());

  ASSERT_EQ(nullptr, nmsv4_node.boxes());
  ASSERT_EQ(nullptr, nmsv4_node.scores());
  ASSERT_EQ(nullptr, nmsv4_node.max_output_size());
  ASSERT_EQ(nullptr, nmsv4_node.iou_threshold());
  ASSERT_EQ(nullptr, nmsv4_node.score_threshold());
}

TEST(CircleNonMaxSuppressionV4Test, input_NEG)
{
  luci::CircleNonMaxSuppressionV4 nmsv4_node;
  luci::CircleNonMaxSuppressionV4 node;

  nmsv4_node.boxes(&node);
  nmsv4_node.scores(&node);
  nmsv4_node.max_output_size(&node);
  nmsv4_node.iou_threshold(&node);
  nmsv4_node.score_threshold(&node);
  ASSERT_NE(nullptr, nmsv4_node.boxes());
  ASSERT_NE(nullptr, nmsv4_node.scores());
  ASSERT_NE(nullptr, nmsv4_node.max_output_size());
  ASSERT_NE(nullptr, nmsv4_node.iou_threshold());
  ASSERT_NE(nullptr, nmsv4_node.score_threshold());

  nmsv4_node.boxes(nullptr);
  nmsv4_node.scores(nullptr);
  nmsv4_node.max_output_size(nullptr);
  nmsv4_node.iou_threshold(nullptr);
  nmsv4_node.score_threshold(nullptr);
  ASSERT_EQ(nullptr, nmsv4_node.boxes());
  ASSERT_EQ(nullptr, nmsv4_node.scores());
  ASSERT_EQ(nullptr, nmsv4_node.max_output_size());
  ASSERT_EQ(nullptr, nmsv4_node.iou_threshold());
  ASSERT_EQ(nullptr, nmsv4_node.score_threshold());
}

TEST(CircleNonMaxSuppressionV4Test, arity_NEG)
{
  luci::CircleNonMaxSuppressionV4 nmsv4_node;

  ASSERT_NO_THROW(nmsv4_node.arg(4));
  ASSERT_THROW(nmsv4_node.arg(5), std::out_of_range);
}

TEST(CircleNonMaxSuppressionV4Test, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleNonMaxSuppressionV4 nmsv4_node;

  TestVisitor tv;
  ASSERT_THROW(nmsv4_node.accept(&tv), std::exception);
}

TEST(CircleNonMaxSuppressionV4Test, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleNonMaxSuppressionV4 nmsv4_node;

  TestVisitor tv;
  ASSERT_THROW(nmsv4_node.accept(&tv), std::exception);
}
