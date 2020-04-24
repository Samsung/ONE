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

  ASSERT_EQ(custom_node.dialect(), luci::CircleDialect::get());
  ASSERT_EQ(custom_node.opcode(), luci::CircleOpcode::CUSTOM);

  ASSERT_EQ(custom_node.arity(), 2);
  ASSERT_EQ(custom_node.arg(0), nullptr);
  ASSERT_EQ(custom_node.arg(1), nullptr);

  ASSERT_EQ(custom_node.numInputs(), 2);
  ASSERT_EQ(custom_node.custom_code().size(), 0);
}

TEST(CircleCustomTest, constructor_NEG) { ASSERT_DEBUG_DEATH(luci::CircleCustom{0}, ""); }

TEST(CircleCustomTest, invalidIndex_NEG)
{
  luci::CircleCustom custom_node(2);

// TODO Fix this not to use '#ifdef'
#ifdef NDEBUG
  // release build will throw
  EXPECT_ANY_THROW(custom_node.arg(5));
#else
  // debug build will fail with assert
  ASSERT_DEBUG_DEATH(custom_node.arg(5), "");
#endif
}
