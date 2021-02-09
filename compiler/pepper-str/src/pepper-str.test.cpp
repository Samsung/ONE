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

#include "pepper/str.h"

#include <iostream>

#include <gtest/gtest.h>

TEST(StrTests, README)
{
  // Let us check whether the example in README.md works!
  int argc = 4;

  std::cout << pepper::str("There are ", argc, " arguments") << std::endl;

  SUCCEED();
}

TEST(StrTests, Empty)
{
  // pepper::str() returns an empty string
  ASSERT_EQ(pepper::str(), "");
}

TEST(StrTests, Single_Int)
{
  // Convert a single "int" value as a string
  ASSERT_EQ(pepper::str(3), "3");
}

TEST(StrTests, Concat_000)
{
  const int n = 3;
  const int m = 4;

  ASSERT_EQ(pepper::str(n, "+", m, "=", n + m), "3+4=7");
}
