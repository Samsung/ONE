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

#include "Strings.h"

#include "luci/Pass/QuantizationParameters.h"

#include <gtest/gtest.h>

TEST(StringsTest, str_to_dtype)
{
  ASSERT_EQ(loco::DataType::U4, luci::str_to_dtype("uint4"));
  ASSERT_EQ(loco::DataType::U8, luci::str_to_dtype("uint8"));
  ASSERT_EQ(loco::DataType::U16, luci::str_to_dtype("uint16"));
  ASSERT_EQ(loco::DataType::U32, luci::str_to_dtype("uint32"));
  ASSERT_EQ(loco::DataType::U64, luci::str_to_dtype("uint64"));

  ASSERT_EQ(loco::DataType::S4, luci::str_to_dtype("int4"));
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

TEST(StringsTest, str_to_granularity)
{
  ASSERT_EQ(luci::QuantizationGranularity::LayerWise, luci::str_to_granularity("layer"));
  ASSERT_EQ(luci::QuantizationGranularity::ChannelWise, luci::str_to_granularity("channel"));

  EXPECT_THROW(luci::str_to_granularity("foo"), std::runtime_error);
}

TEST(StringsTest, str_vec_to_dtype_vec)
{
  std::vector<std::string> input1 = {"uint8", "int16", "float32"};
  auto result1 = luci::str_vec_to_dtype_vec(input1);
  ASSERT_EQ(3, result1.size());
  ASSERT_EQ(loco::DataType::U8, result1[0]);
  ASSERT_EQ(loco::DataType::S16, result1[1]);
  ASSERT_EQ(loco::DataType::FLOAT32, result1[2]);

  std::vector<std::string> input2 = {"uint8", "int16", "float32", ""};
  auto result2 = luci::str_vec_to_dtype_vec(input2);
  ASSERT_EQ(4, result2.size());
  ASSERT_EQ(loco::DataType::U8, result2[0]);
  ASSERT_EQ(loco::DataType::S16, result2[1]);
  ASSERT_EQ(loco::DataType::FLOAT32, result2[2]);
  ASSERT_EQ(loco::DataType::Unknown, result2[3]);

  std::vector<std::string> input3 = {"uint8"};
  auto result3 = luci::str_vec_to_dtype_vec(input3);
  ASSERT_EQ(1, result3.size());
  ASSERT_EQ(loco::DataType::U8, result3[0]);
}
