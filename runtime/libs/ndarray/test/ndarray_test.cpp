/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"

#include "ndarray/Array.h"

using namespace ndarray;

TEST(NDArray_tests, basic_data_test)
{

  float raw_data[] = {1, 2, 3, 4};

  Array<float> data22{raw_data, {2, 2}};

  ASSERT_FLOAT_EQ(data22.at(0, 0), 1);
  ASSERT_FLOAT_EQ(data22.at(0, 1), 2);
  ASSERT_FLOAT_EQ(data22.at(1, 0), 3);
  ASSERT_FLOAT_EQ(data22.at(1, 1), 4);

  Array<float> data14{raw_data, {1, 4}};
  ASSERT_FLOAT_EQ(data22.at(0, 0), 1);
  ASSERT_FLOAT_EQ(data22.at(0, 1), 2);
  ASSERT_FLOAT_EQ(data22.at(0, 2), 3);
  ASSERT_FLOAT_EQ(data22.at(0, 3), 4);
}

TEST(NDArray_tests, slice_write_test)
{
  float raw_data[4] = {0};

  Array<float> data22{raw_data, {2, 2}};

  data22.slice(1) = {1, 2};

  ASSERT_FLOAT_EQ(data22.at(0, 0), 0);
  ASSERT_FLOAT_EQ(data22.at(0, 1), 0);
  ASSERT_FLOAT_EQ(data22.at(1, 0), 1);
  ASSERT_FLOAT_EQ(data22.at(1, 1), 2);
}

TEST(NDArray_tests, slice_read_test)
{
  float raw_data[4] = {1, 2, 3, 4};

  Array<float> data22{raw_data, {2, 2}};

  auto slice = data22.slice(1);

  ASSERT_FLOAT_EQ(slice[0], 3);
  ASSERT_FLOAT_EQ(slice[1], 4);
}

TEST(NDArray_tests, multidim_test)
{
  float raw_data[5] = {0, 1, 2, 3, 4};

  Array<float> data22{raw_data, {1, 1, 1, 1, 5}};

  ASSERT_FLOAT_EQ(data22.at(0, 0, 0, 0, 0), 0);
  ASSERT_FLOAT_EQ(data22.at(0, 0, 0, 0, 1), 1);
  ASSERT_FLOAT_EQ(data22.at(0, 0, 0, 0, 2), 2);
  ASSERT_FLOAT_EQ(data22.at(0, 0, 0, 0, 3), 3);
  ASSERT_FLOAT_EQ(data22.at(0, 0, 0, 0, 4), 4);
}

TEST(NDArray_tests, slice_assign_test)
{
  std::vector<float> v1{1, 2, 3, 4, 5};
  std::vector<float> v2(5);

  ContiguousSpan<float> span1(v1.begin(), v1.end());
  ContiguousSpan<float> span2(v2.begin(), v2.end());

  span2.assign(span1);

  ASSERT_EQ(v1, v2);
}
