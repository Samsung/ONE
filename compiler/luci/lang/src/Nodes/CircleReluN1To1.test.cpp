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

#include "luci/IR/Nodes/CircleReluN1To1.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleReluN1To1Test, constructor)
{
  luci::CircleReluN1To1 relu_node;

  ASSERT_EQ(luci::CircleDialect::get(), relu_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::RELU_N1_TO_1, relu_node.opcode());

  ASSERT_EQ(nullptr, relu_node.features());
}
