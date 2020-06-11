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

#include "luci/IR/Nodes/CircleLogistic.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleLogisticTest, constructor)
{
  luci::CircleLogistic logistic_node;

  ASSERT_EQ(luci::CircleDialect::get(), logistic_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::LOGISTIC, logistic_node.opcode());

  ASSERT_EQ(nullptr, logistic_node.x());
}

TEST(CircleLogisticTest, input_NEG)
{
  luci::CircleLogistic logistic_node;
  luci::CircleLogistic node;

  logistic_node.x(&node);
  ASSERT_NE(nullptr, logistic_node.x());

  logistic_node.x(nullptr);
  ASSERT_EQ(nullptr, logistic_node.x());
}

TEST(CircleLogisticTest, arity_NEG)
{
  luci::CircleLogistic logistic_node;

  ASSERT_NO_THROW(logistic_node.arg(0));
  ASSERT_THROW(logistic_node.arg(1), std::out_of_range);
}

TEST(CircleLogisticTest, visit_mutable_NEG)
{
  struct TestVisitor final : public luci::CircleNodeMutableVisitor<void>
  {
  };

  luci::CircleLogistic logistic_node;

  TestVisitor tv;
  ASSERT_THROW(logistic_node.accept(&tv), std::exception);
}

TEST(CircleLogisticTest, visit_NEG)
{
  struct TestVisitor final : public luci::CircleNodeVisitor<void>
  {
  };

  luci::CircleLogistic logistic_node;

  TestVisitor tv;
  ASSERT_THROW(logistic_node.accept(&tv), std::exception);
}
