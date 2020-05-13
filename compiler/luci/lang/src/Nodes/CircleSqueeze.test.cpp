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

#include "luci/IR/Nodes/CircleSqueeze.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleSqueezeTest, constructor_P)
{
  luci::CircleSqueeze squeeze;

  ASSERT_EQ(luci::CircleDialect::get(), squeeze.dialect());
  ASSERT_EQ(luci::CircleOpcode::SQUEEZE, squeeze.opcode());

  ASSERT_EQ(nullptr, squeeze.input());
  ASSERT_EQ(0, squeeze.squeeze_dims().size());
}

TEST(CircleSqueezeTest, squeeze_dims)
{
  luci::CircleSqueeze squeeze;

  squeeze.squeeze_dims({1, 2});

  ASSERT_EQ(1, squeeze.squeeze_dims().at(0));
  ASSERT_EQ(2, squeeze.squeeze_dims().at(1));
}
