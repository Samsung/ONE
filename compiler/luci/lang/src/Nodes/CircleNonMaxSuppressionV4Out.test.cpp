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

#include "luci/IR/Nodes/CircleNonMaxSuppressionV4Out.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleNonMaxSuppressionV4OutTest, constructor)
{
  luci::CircleNonMaxSuppressionV4Out vout_node;

  ASSERT_EQ(luci::CircleDialect::get(), vout_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::CIRCLENONMAXSUPPRESSIONV4OUT, vout_node.opcode());

  ASSERT_EQ(nullptr, vout_node.input());
  ASSERT_EQ(-1, vout_node.index());
}
