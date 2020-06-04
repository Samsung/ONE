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

#include "luci/IR/Nodes/CircleAveragePool2D.h"

#include "luci/IR/CircleDialect.h"
#include <gtest/gtest.h>

TEST(CircleAveragePool2DTest, constructor_P)
{
  luci::CircleAveragePool2D average_pool_2d_node;

  ASSERT_EQ(luci::CircleDialect::get(), average_pool_2d_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::AVERAGE_POOL_2D, average_pool_2d_node.opcode());

  ASSERT_EQ(nullptr, average_pool_2d_node.value());
  ASSERT_EQ(luci::Padding::UNDEFINED, average_pool_2d_node.padding());
  ASSERT_EQ(1, average_pool_2d_node.filter()->h());
  ASSERT_EQ(1, average_pool_2d_node.filter()->w());
  ASSERT_EQ(1, average_pool_2d_node.stride()->h());
  ASSERT_EQ(1, average_pool_2d_node.stride()->w());
}
