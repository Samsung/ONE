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

#include "luci/IR/Nodes/CircleSpaceToBatchND.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleSpaceToBatchNDTest, constructor)
{
  luci::CircleSpaceToBatchND stb_node;

  ASSERT_EQ(luci::CircleDialect::get(), stb_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::SPACE_TO_BATCH_ND, stb_node.opcode());

  ASSERT_EQ(nullptr, stb_node.input());
  ASSERT_EQ(nullptr, stb_node.block_shape());
  ASSERT_EQ(nullptr, stb_node.paddings());
}
