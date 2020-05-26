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

#include "luci/IR/Nodes/CircleLocalResponseNormalization.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleLocalResponseNormalizationTest, constructor_P)
{
  luci::CircleLocalResponseNormalization local_response_normalization_node;

  ASSERT_EQ(luci::CircleDialect::get(), local_response_normalization_node.dialect());
  ASSERT_EQ(luci::CircleOpcode::LOCAL_RESPONSE_NORMALIZATION,
            local_response_normalization_node.opcode());

  ASSERT_EQ(nullptr, local_response_normalization_node.input());
  ASSERT_EQ(5, local_response_normalization_node.radius());
  ASSERT_EQ(1.0f, local_response_normalization_node.bias());
  ASSERT_EQ(1.0f, local_response_normalization_node.alpha());
  ASSERT_EQ(0.5f, local_response_normalization_node.beta());
}
