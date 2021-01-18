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

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

#include "TypeMapper.h"

TEST(TypeMapperTest, simple_test)
{
  EXPECT_EQ(loco::DataType::FLOAT32, luci::TypeMapper<float>::get());
  EXPECT_EQ(loco::DataType::U8, luci::TypeMapper<uint8_t>::get());
  EXPECT_EQ(loco::DataType::S16, luci::TypeMapper<int16_t>::get());
}

TEST(TypeMapperTest, wrong_condition_NEG)
{
  EXPECT_EQ(loco::DataType::Unknown, luci::TypeMapper<double>::get());
}
