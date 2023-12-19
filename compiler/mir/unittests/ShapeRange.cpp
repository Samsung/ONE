/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "gtest/gtest.h"
#include "mir/ShapeRange.h"

using namespace mir;

namespace
{

struct ParamType
{
  int32_t actual_length;
  Shape shape;

  template <typename... Args>
  explicit ParamType(int32_t actual_len, Args &&...args)
    : actual_length(actual_len), shape({static_cast<int32_t>(args)...})
  {
  }
};

class ShapeIteratorTest : public ::testing::TestWithParam<ParamType>
{
};

TEST_P(ShapeIteratorTest, ElementCount)
{
  Shape sh(GetParam().shape);
  ShapeRange r(sh);

  int32_t cnt = 0;
  for (auto &idx : r)
  {
    (void)idx;
    cnt++;
  }

  ASSERT_EQ(cnt, GetParam().actual_length);
}

std::vector<ParamType> test_data{ParamType{6, 1, 2, 3}, ParamType{16, 2, 2, 4},
                                 ParamType{1, 1, 1, 1, 1, 1}, ParamType{5, 5, 1, 1, 1, 1, 1}};

INSTANTIATE_TEST_SUITE_P(SimpleInput, ShapeIteratorTest, ::testing::ValuesIn(test_data));

TEST(ShapeRange, Contains)
{
  const int h = 2;
  const int w = 3;
  Shape shape{static_cast<int32_t>(h), static_cast<int32_t>(w)};
  ShapeRange range(shape);
  Index index{0, 0, 0, 0};
  for (int32_t row = -2; row < h + 1; ++row)
    for (int32_t col = -2; col < w + 1; ++col)
    {
      Index idx{row, col};
      if (row < 0 || row >= h || col < 0 || col >= w)
        ASSERT_FALSE(range.contains(idx));
      else
        ASSERT_TRUE(range.contains(idx));
    }
}
} // namespace
