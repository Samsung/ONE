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

#include "luci/IR/Nodes/CircleMaxPool2D.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleMaxPool2DTest, constructor_P)
{
  luci::CircleMaxPool2D maxpool2d_node;

  ASSERT_EQ(luci::CircleDialect::get(), maxpool2d_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::MAX_POOL_2D, maxpool2d_node.opcode());

  ASSERT_EQ(nullptr, maxpool2d_node.value());
  ASSERT_NE(maxpool2d_node.filter(), nullptr);
  ASSERT_NE(maxpool2d_node.stride(), nullptr);
}
