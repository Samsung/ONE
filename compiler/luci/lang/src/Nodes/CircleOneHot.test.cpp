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

#include "luci/IR/Nodes/CircleOneHot.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleGatherTest, constructor)
{
  luci::CircleOneHot one_hot_node;

  ASSERT_EQ(one_hot_node.dialect(), luci::CircleDialect::get());
  ASSERT_EQ(one_hot_node.opcode(), luci::CircleOpcode::ONE_HOT);

  ASSERT_EQ(one_hot_node.indices(), nullptr);
  ASSERT_EQ(one_hot_node.depth(), nullptr);
  ASSERT_EQ(one_hot_node.on_value(), nullptr);
  ASSERT_EQ(one_hot_node.off_value(), nullptr);
  ASSERT_EQ(one_hot_node.axis(), -1);
}
