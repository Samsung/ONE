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

#include "coco/ADT/Span.h"

#include <gtest/gtest.h>

TEST(SpanTest, constructor)
{
  const uint32_t arr_size = 16;
  int arr_data[arr_size];

  coco::Span<int> span{arr_data, arr_size};

  coco::Span<int> &ref = span;
  const coco::Span<int> &cref = span;

  ASSERT_EQ(ref.data(), arr_data);
  ASSERT_EQ(cref.data(), arr_data);
  ASSERT_EQ(ref.size(), arr_size);
}

TEST(SpanTest, array_subscript_operator)
{
  // Create a stack-allocated chunk
  const uint32_t arr_size = 16;
  int arr_data[arr_size];

  for (uint32_t n = 0; n < arr_size; ++n)
  {
    arr_data[n] = n;
  }

  // Create a Span
  coco::Span<int> span{arr_data, arr_size};

  coco::Span<int> &ref = span;
  const coco::Span<int> &cref = span;

  ASSERT_EQ(ref[3], 3);
  ASSERT_EQ(cref[3], 3);

  arr_data[3] = 16;

  ASSERT_EQ(ref[3], 16);
  ASSERT_EQ(cref[3], 16);
}
