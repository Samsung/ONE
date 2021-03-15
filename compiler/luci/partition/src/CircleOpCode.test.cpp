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

#include "CircleOpCode.h"

// NOTE any node will do for testing
#include <luci/IR/Nodes/CircleSqrt.h>

#include <gtest/gtest.h>

TEST(CircleOpCodeTest, name)
{
  auto g = loco::make_graph();
  auto node = g->nodes()->create<luci::CircleSqrt>();

  auto name = luci::opcode_name(node);
  ASSERT_EQ(name, "SQRT");
}
