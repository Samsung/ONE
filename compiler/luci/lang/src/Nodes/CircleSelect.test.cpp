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

#include "luci/IR/Nodes/CircleSelect.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleSelectTest, constructor)
{
  luci::CircleSelect select_node;

  ASSERT_EQ(luci::CircleDialect::get(), select_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SELECT, select_node.opcode());

  ASSERT_EQ(nullptr, select_node.condition());
  ASSERT_EQ(nullptr, select_node.t());
  ASSERT_EQ(nullptr, select_node.e());
}
