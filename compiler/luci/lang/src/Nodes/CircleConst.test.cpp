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

#include "luci/IR/Nodes/CircleConst.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleConstTest, constructor)
{
  luci::CircleConst const_node;

  ASSERT_EQ(luci::CircleDialect::get(), const_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CIRCLECONST, const_node.opcode());
}

TEST(CircleConstTest, dype_size)
{
  luci::CircleConst const_node;

  const_node.dtype(loco::DataType::S32);
  const_node.size<loco::DataType::S32>(1);

  ASSERT_EQ(loco::DataType::S32, const_node.dtype());
  ASSERT_EQ(1, const_node.size<loco::DataType::S32>());
}

TEST(CircleConstTest, scalar)
{
  luci::CircleConst const_node;

  const_node.dtype(loco::DataType::S32);
  const_node.size<loco::DataType::S32>(1);
  const_node.scalar<loco::DataType::S32>() = 1;

  auto const &cs = const_node.scalar<loco::DataType::S32>();
  ASSERT_EQ(1, cs);
}

TEST(CircleConstTest, string)
{
  luci::CircleConst const_node;

  const_node.dtype(loco::DataType::STRING);
  const_node.size<loco::DataType::STRING>(1);
  const_node.at<loco::DataType::STRING>(0) = std::string("Hello");

  ASSERT_EQ(loco::DataType::STRING, const_node.dtype());
  ASSERT_EQ(1, const_node.size<loco::DataType::STRING>());
  EXPECT_TRUE(std::string("Hello") == const_node.at<loco::DataType::STRING>(0));
}
