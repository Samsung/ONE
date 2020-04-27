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

  ASSERT_EQ(luci::CircleDialect::get(), instance_norm.dialect());
  ASSERT_EQ(luci::CircleOpcode::INSTANCE_NORM, instance_norm.opcode());

  ASSERT_EQ(nullptr, instance_norm.input());
  ASSERT_EQ(nullptr, instance_norm.gamma());
  ASSERT_EQ(nullptr, instance_norm.beta());
  ASSERT_FLOAT_EQ(instance_norm.epsilon(), 1e-05);
  ASSERT_EQ(luci::FusedActFunc::UNDEFINED, instance_norm.fusedActivationFunction());
}
