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

#include "CircleOptimizerUtils.h"

#include "luci/Pass/QuantizationParameters.h"

#include <gtest/gtest.h>

TEST(CircleOptimizerUtilTest, str_to_dtype)
{
  ASSERT_EQ(loco::DataType::U8, luci::str_to_dtype("uint8"));
  ASSERT_EQ(loco::DataType::U16, luci::str_to_dtype("uint16"));
  ASSERT_EQ(loco::DataType::U32, luci::str_to_dtype("uint32"));
  ASSERT_EQ(loco::DataType::U64, luci::str_to_dtype("uint64"));

  ASSERT_EQ(loco::DataType::S8, luci::str_to_dtype("int8"));
  ASSERT_EQ(loco::DataType::S16, luci::str_to_dtype("int16"));
  ASSERT_EQ(loco::DataType::S32, luci::str_to_dtype("int32"));
  ASSERT_EQ(loco::DataType::S64, luci::str_to_dtype("int64"));

  ASSERT_EQ(loco::DataType::FLOAT16, luci::str_to_dtype("float16"));
  ASSERT_EQ(loco::DataType::FLOAT32, luci::str_to_dtype("float32"));
  ASSERT_EQ(loco::DataType::FLOAT64, luci::str_to_dtype("float64"));

  ASSERT_EQ(loco::DataType::BOOL, luci::str_to_dtype("bool"));

  ASSERT_EQ(loco::DataType::Unknown, luci::str_to_dtype("foo"));
}

TEST(CircleOptimizerUtilTest, str_to_granularity)
{
  ASSERT_EQ(luci::QuantizationGranularity::LayerWise, luci::str_to_granularity("layer"));
  ASSERT_EQ(luci::QuantizationGranularity::ChannelWise, luci::str_to_granularity("channel"));

  EXPECT_THROW(luci::str_to_granularity("foo"), std::runtime_error);
}
