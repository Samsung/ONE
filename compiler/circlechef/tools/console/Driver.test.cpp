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

#include <gtest/gtest.h>

// entry function to test from Driver.cpp
int entry_stream(std::istream &is);

TEST(CircleChefDriverTest, entry_empty_NEG)
{
  std::istringstream empty_input("");

  ASSERT_EQ(0, entry_stream(empty_input));
}

TEST(CircleChefDriverTest, entry_invaid_NEG)
{
  std::istringstream empty_input("invalid: input");

  ASSERT_NE(0, entry_stream(empty_input));
}

TEST(CircleChefDriverTest, entry_invaid_version_NEG)
{
  std::istringstream empty_input("version: 9999");

  ASSERT_NE(0, entry_stream(empty_input));
}
