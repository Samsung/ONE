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

#include "Driver.test.h"

#include <gtest/gtest.h>

TEST(DriverTest, NoArg_NEG)
{
  Argv<1> argv;
  argv.add("circle-opselector");

  ::testing::internal::CaptureStderr();
  ::testing::internal::CaptureStdout();
  int result = entry(1, argv.argv());
  ::testing::internal::GetCapturedStdout();
  ASSERT_EQ(EXIT_FAILURE, result);
}

TEST(DriverTest, Wrong_ID)
{
  std::string str1 = "1";
  std::string empty = "";
  std::string no_integer = "1531538X5";

  ASSERT_EQ(true, is_number(str1));
  ASSERT_EQ(false, is_number(empty));
  ASSERT_EQ(false, is_number(no_integer));
}

TEST(DriverTest, Split)
{
  std::vector<uint32_t> vec1;
  std::vector<uint32_t> vec2;

  std::string hyphen = "1-3,8-10";
  std::string comma = "1,2,3";

  vec1.push_back(1);
  vec1.push_back(2);
  vec1.push_back(3);
  vec1.push_back(8);
  vec1.push_back(9);
  vec1.push_back(10);

  vec2.push_back(1);
  vec2.push_back(2);
  vec2.push_back(3);

  ASSERT_EQ(vec1, split_id_input(hyphen));
  ASSERT_EQ(vec2, split_id_input(comma));
}
