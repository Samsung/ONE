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

#include "luci/IR/Nodes/CircleInstanceNorm.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleInstanceNormTest, constructor)
{
  luci::CircleInstanceNorm instance_norm;

  ASSERT_EQ(instance_norm.dialect(), luci::CircleDialect::get());
  ASSERT_EQ(instance_norm.opcode(), luci::CircleOpcode::INSTANCE_NORM);

  ASSERT_EQ(instance_norm.input(), nullptr);
  ASSERT_EQ(instance_norm.gamma(), nullptr);
  ASSERT_EQ(instance_norm.beta(), nullptr);
  ASSERT_FLOAT_EQ(instance_norm.epsilon(), 1e-05);
  ASSERT_EQ(instance_norm.fusedActivationFunction(), luci::FusedActFunc::UNDEFINED);
}
