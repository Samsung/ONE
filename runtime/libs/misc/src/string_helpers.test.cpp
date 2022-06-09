/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "misc/string_helpers.h"

#include <gtest/gtest.h>

TEST(StringHelpersTest, split)
{
  const std::string example = "abc;def;ghi";

  auto str_vector = nnfw::misc::split(example, ';');

  ASSERT_EQ(str_vector.size(), 3);
  EXPECT_STREQ(str_vector[0].c_str(), "abc");
  EXPECT_STREQ(str_vector[1].c_str(), "def");
  EXPECT_STREQ(str_vector[2].c_str(), "ghi");
}

TEST(StringHelpersTest, neg_split_empty)
{
  const std::string example = "";

  auto str_vector = nnfw::misc::split(example, ';');

  ASSERT_EQ(str_vector.size(), 0);
}

TEST(StringHelpersTest, neg_nonsplit)
{
  const std::string example = "abc;def;ghi";

  auto str_vector = nnfw::misc::split(example, ':');

  ASSERT_EQ(str_vector.size(), 1);
  EXPECT_STREQ(str_vector[0].c_str(), example.c_str());
}

TEST(StringHelpersTest, append)
{
  auto append_str = nnfw::misc::str("abc", "-", 1);

  EXPECT_STREQ(append_str.c_str(), "abc-1");
}

TEST(StringHelpersTest, neg_append_nullstr)
{
  const char *null_str = nullptr;
  auto append_str = nnfw::misc::str(null_str, null_str);

  ASSERT_EQ(append_str.size(), 0);
}

TEST(StringHelpersTest, join)
{
  const std::vector<std::string> example = {"abc", "def", "ghi"};

  auto join_str = nnfw::misc::join(example.begin(), example.end(), ";");
  EXPECT_STREQ(join_str.c_str(), "abc;def;ghi");
}

TEST(StringHelpersTest, neg_join_empty)
{
  const std::vector<std::string> example = {};

  auto join_str = nnfw::misc::join(example.begin(), example.end(), ";");
  ASSERT_EQ(join_str.size(), 0);
}
