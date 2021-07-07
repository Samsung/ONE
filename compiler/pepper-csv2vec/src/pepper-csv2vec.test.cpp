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

#include "pepper/csv2vec.h"

#include <gtest/gtest.h>

TEST(csv2vec, simple_string)
{
  auto ret = pepper::csv_to_vector<std::string>("hello,world");

  ASSERT_EQ(2, ret.size());
  ASSERT_TRUE("hello" == ret.at(0));
  ASSERT_TRUE("world" == ret.at(1));
}

TEST(csv2vec, simple_int32)
{
  auto ret = pepper::csv_to_vector<int32_t>("1,2,3");

  ASSERT_EQ(3, ret.size());
  ASSERT_EQ(1, ret.at(0));
  ASSERT_EQ(3, ret.at(2));
}

TEST(csv2vec, is_one_of)
{
  auto ret = pepper::csv_to_vector<std::string>("hello,world");

  ASSERT_TRUE(pepper::is_one_of<std::string>("hello", ret));
  ASSERT_FALSE(pepper::is_one_of<std::string>("good", ret));
}

TEST(csv2vec, empty_string_NEG)
{
  // should not abort
  EXPECT_NO_THROW(pepper::csv_to_vector<std::string>(""));
}

TEST(csv2vec, invalid_int32_NEG)
{
  auto ret = pepper::csv_to_vector<int32_t>("hello,world");

  ASSERT_EQ(0, ret.size());
}
