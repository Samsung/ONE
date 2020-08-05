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

#include "luci/IR/Nodes/CircleCustom.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleCustomTest, constructor)
{
  luci::CircleCustom custom_node(2);

  ASSERT_EQ(luci::CircleDialect::get(), custom_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CUSTOM, custom_node.opcode());

  ASSERT_EQ(2, custom_node.arity());
  ASSERT_EQ(nullptr, custom_node.arg(0));
  ASSERT_EQ(nullptr, custom_node.arg(1));

  ASSERT_EQ(2, custom_node.numInputs());
  ASSERT_EQ(0, custom_node.custom_code().size());
}

TEST(CircleCustomTest, constructor_NEG)
{
  ASSERT_DEBUG_DEATH(luci::CircleCustom{0}, "");

  SUCCEED();
}

TEST(CircleCustomTest, invalidIndex_NEG)
{
  luci::CircleCustom custom_node(2);

  EXPECT_ANY_THROW(custom_node.arg(5));
}
