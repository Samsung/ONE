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

#include "luci/IR/Nodes/CircleFill.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleFillTest, constructor_P)
{
  luci::CircleFill fill;

  ASSERT_EQ(fill.dialect(), luci::CircleDialect::get());
  ASSERT_EQ(fill.opcode(), luci::CircleOpcode::FILL);

  ASSERT_EQ(nullptr, fill.dims());
  ASSERT_EQ(nullptr, fill.value());
}
