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

#include "Weight.h"

#include <gtest/gtest.h>

TEST(ANN_IR_WEIGHT, constructor)
{
  ann::Weight weight;

  ASSERT_EQ(weight.base(), nullptr);
  ASSERT_EQ(weight.size(), 0);
}

TEST(ANN_IR_WEIGHT, fill_scalar_int)
{
  ann::Weight weight;

  weight.fill(3);

  ASSERT_NE(weight.base(), nullptr);
  ASSERT_EQ(*reinterpret_cast<const int *>(weight.base()), 3);
}

TEST(ANN_IR_WEIGHT, fill_vector_float)
{
  std::vector<float> values{1.0f, 2.0f};

  ann::Weight weight;

  weight.fill(values.begin(), values.end());

  ASSERT_NE(weight.base(), nullptr);

  auto arr = reinterpret_cast<const float *>(weight.base());

  ASSERT_FLOAT_EQ(arr[0], 1.0f);
  ASSERT_FLOAT_EQ(arr[1], 2.0f);
}
