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

#include "luci/Import/CircleReader.h"

#include <gtest/gtest.h>

TEST(VectorWrapperTest, basic_pattern)
{
  auto fb_builder = flatbuffers::FlatBufferBuilder();

  const std::vector<int32_t> data = {1, 4, 2, 0, 7};
  auto const vec_offset = fb_builder.CreateVector(data.data(), data.size());
  auto const vec_pointer = GetTemporaryPointer(fb_builder, vec_offset);

  auto const wrapper = luci::wrap(vec_pointer);

  ASSERT_EQ(wrapper.size(), data.size());
  ASSERT_TRUE(std::equal(wrapper.begin(), wrapper.end(), data.begin()));
}

TEST(VectorWrapperTest, wrong_data_NEG)
{
  auto fb_builder = flatbuffers::FlatBufferBuilder();

  std::vector<int32_t> data = {1, 4, 2, 0, 7};
  auto const vec_offset = fb_builder.CreateVector(data.data(), data.size());
  auto const vec_pointer = GetTemporaryPointer(fb_builder, vec_offset);

  auto const wrapper = luci::wrap(vec_pointer);

  // change data
  std::reverse(data.begin(), data.end());

  ASSERT_EQ(wrapper.size(), data.size());
  ASSERT_FALSE(std::equal(wrapper.begin(), wrapper.end(), data.begin()));
}

TEST(VectorWrapperTest, null_pointer)
{
  flatbuffers::Vector<int32_t> *vec_pointer = nullptr;
  auto const wrapper = luci::wrap(vec_pointer);

  ASSERT_TRUE(wrapper.null());
  ASSERT_TRUE(wrapper.empty());
}

TEST(VectorWrapperTest, prohibited_access_NEG)
{
  flatbuffers::Vector<uint8_t> *vec_pointer = nullptr;
  auto const wrapper = luci::wrap(vec_pointer);

  ASSERT_ANY_THROW(wrapper.at(0));
}
