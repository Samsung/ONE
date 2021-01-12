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

#include "luci/IR/Nodes/CircleFakeQuant.h"

#include "luci/IR/CircleDialect.h"
#include "luci/IR/CircleNodeVisitor.h"

#include <gtest/gtest.h>

TEST(CircleFakeQuantTest, constructor_P)
{
  luci::CircleFakeQuant fakequant;

  ASSERT_EQ(fakequant.dialect(), luci::CircleDialect::get());
  ASSERT_EQ(fakequant.opcode(), luci::CircleOpcode::FAKE_QUANT);

  ASSERT_EQ(nullptr, fakequant.inputs());
  ASSERT_EQ(0.0f, fakequant.min());
  ASSERT_EQ(0.0f, fakequant.max());
  ASSERT_EQ(0, fakequant.num_bits());
  ASSERT_FALSE(fakequant.narrow_range());
}
