/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleNodes.h"

#include "CircleDialect.h"
#include "CircleOpcode.h"

#include <gtest/gtest.h>

TEST(CircleInstanceNormTest, constructor)
{
  locoex::CircleInstanceNorm instance_norm;

  ASSERT_EQ(locoex::CircleDialect::get(), instance_norm.dialect());
  ASSERT_EQ(locoex::CircleOpcode::INSTANCE_NORM, instance_norm.opcode());

  ASSERT_EQ(nullptr, instance_norm.input());
  ASSERT_EQ(nullptr, instance_norm.gamma());
  ASSERT_EQ(nullptr, instance_norm.beta());
  ASSERT_FLOAT_EQ(1e-05, instance_norm.epsilon());
  ASSERT_EQ(locoex::FusedActFunc::UNDEFINED, instance_norm.fusedActivationFunction());
}
