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

#include "fipe.h"

#include <string>

#include <gtest/gtest.h>

namespace
{

int dec(int n) { return n - 1; }

} // namespace

TEST(FunctionPipeTests, top_level_function)
{
  // GCC rejects this code if dec is not wrapped by "fipe::wrap"
  // TODO Find a better way
  ASSERT_EQ(4 | fipe::wrap(dec), 3);
}

TEST(FunctionPipeTests, static_method)
{
  struct Sample
  {
    static int dbl(int n) { return n * 2; }
  };

  ASSERT_EQ(4 | fipe::wrap(Sample::dbl), 8);
}

TEST(FunctionPipeTests, normal_method)
{
  struct Sample
  {
  public:
    int shift(int n) { return n + shiftamt; }

  private:
    int shiftamt = 6;
  };

  using namespace std::placeholders;

  Sample s;

  auto value = 4 | std::bind(&Sample::shift, &s, _1);

  ASSERT_EQ(value, 10);
}

TEST(FunctionPipeTests, lambda)
{
  auto inc = [](int n) { return n + 1; };
  ASSERT_EQ(4 | inc, 5);
}

TEST(FunctionPipeTests, functor) { ASSERT_EQ(4 | std::negate<int>(), -4); }
