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

#include "luci/IR/Nodes/CircleConstReference.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleConstReferenceTest, constructor)
{
  luci::CircleConstReference node;

  ASSERT_EQ(luci::CircleDialect::get(), node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CIRCLECONSTREFERENCE, node.opcode());
}

TEST(CircleConstReferenceTest, value_setting)
{
  luci::CircleConstReference node;
  node.dtype(loco::DataType::S32);

  const std::vector<int32_t> data = {1, 3, 2};
  node.bind_buffer(reinterpret_cast<const uint8_t *>(data.data()), data.size() * sizeof(int32_t));

  ASSERT_EQ(loco::DataType::S32, node.dtype());
  ASSERT_EQ(reinterpret_cast<const int32_t *>(node.data()), data.data());
  ASSERT_EQ(node.at<loco::DataType::S32>(0), data[0]);
  ASSERT_EQ(node.at<loco::DataType::S32>(1), data[1]);
  ASSERT_EQ(node.at<loco::DataType::S32>(2), data[2]);
}

TEST(CircleConstReferenceTest, scalar)
{
  luci::CircleConstReference node;
  node.dtype(loco::DataType::S32);

  const int32_t scalar = 7;
  node.bind_buffer(reinterpret_cast<const uint8_t *>(&scalar), sizeof(int32_t));

  ASSERT_EQ(loco::DataType::S32, node.dtype());
  ASSERT_EQ(node.scalar<loco::DataType::S32>(), scalar);
}
