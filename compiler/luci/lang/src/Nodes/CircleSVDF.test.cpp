/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleSVDF.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleSVDFTest, constructor)
{
  luci::CircleSVDF svdf_node;

  ASSERT_EQ(luci::CircleDialect::get(), svdf_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SVDF, svdf_node.opcode());

  ASSERT_EQ(nullptr, svdf_node.input());
  ASSERT_EQ(nullptr, svdf_node.weight_feature());
  ASSERT_EQ(nullptr, svdf_node.weight_time());
  ASSERT_EQ(nullptr, svdf_node.bias());
  ASSERT_EQ(nullptr, svdf_node.input_activation_state());

  ASSERT_EQ(false, svdf_node.asymmetric_quantize_inputs());
  ASSERT_EQ(0, svdf_node.svdf_rank());
}

TEST(CircleSVDFTest, input_NEG)
{
  luci::CircleSVDF svdf_node;
  luci::CircleSVDF node;

  svdf_node.input(&node);
  svdf_node.weight_feature(&node);
  svdf_node.weight_time(&node);
  svdf_node.bias(&node);
  svdf_node.input_activation_state(&node);

  ASSERT_NE(nullptr, svdf_node.input());
  ASSERT_NE(nullptr, svdf_node.weight_feature());
  ASSERT_NE(nullptr, svdf_node.weight_time());
  ASSERT_NE(nullptr, svdf_node.bias());
  ASSERT_NE(nullptr, svdf_node.input_activation_state());

  svdf_node.input(nullptr);
  svdf_node.weight_feature(nullptr);
  svdf_node.weight_time(nullptr);
  svdf_node.bias(nullptr);
  svdf_node.input_activation_state(nullptr);

  ASSERT_EQ(nullptr, svdf_node.input());
  ASSERT_EQ(nullptr, svdf_node.weight_feature());
  ASSERT_EQ(nullptr, svdf_node.weight_time());
  ASSERT_EQ(nullptr, svdf_node.bias());
  ASSERT_EQ(nullptr, svdf_node.input_activation_state());
}

TEST(CircleSVDFTest, arity_NEG)
{
  luci::CircleSVDF svdf_node;

  ASSERT_NO_THROW(svdf_node.arg(4));
  ASSERT_THROW(svdf_node.arg(5), std::out_of_range);
}

TEST(CircleSVDFTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleSVDF svdf_node;

  TestVisitor tv;
  ASSERT_THROW(svdf_node.accept(&tv), std::exception);
}

TEST(CircleSVDFTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleSVDF svdf_node;

  TestVisitor tv;
  ASSERT_THROW(svdf_node.accept(&tv), std::exception);
}
