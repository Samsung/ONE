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
  int32_t raw_data_int[] = {1, 2, 3, 4};

  Array<float> data22{raw_data, {2, 2}};
  Array<int32_t> data22_int{raw_data_int, {2, 2}};

  ASSERT_FLOAT_EQ(data22.at(0, 0), 1);
  ASSERT_FLOAT_EQ(data22.at(0, 1), 2);
  ASSERT_FLOAT_EQ(data22.at(1, 0), 3);
  ASSERT_FLOAT_EQ(data22.at(1, 1), 4);
  ASSERT_EQ(data22.shape().rank(), 2);
  ASSERT_EQ(data22.shape().dim(0), 2);
  ASSERT_EQ(data22.shape().dim(1), 2);

  Array<float> data14{raw_data, {1, 4}};
  ASSERT_FLOAT_EQ(data14.at(0, 0), 1);
  ASSERT_FLOAT_EQ(data14.at(0, 1), 2);
  ASSERT_FLOAT_EQ(data14.at(0, 2), 3);
  ASSERT_FLOAT_EQ(data14.at(0, 3), 4);
  ASSERT_EQ(data14.shape().rank(), 2);
  ASSERT_EQ(data14.shape().dim(0), 1);
  ASSERT_EQ(data14.shape().dim(1), 4);

  // <float, false>
  {
    ContiguousSpan<float> cs = data22.flat();
    ASSERT_EQ(cs.size(), 4);
    ASSERT_FLOAT_EQ(cs.at(3), 4);
  }

  // <float, true>
  {
    ContiguousSpan<float, true> cs_const = data22.flat();
    ASSERT_EQ(cs_const.size(), 4);
    ASSERT_FLOAT_EQ(cs_const.at(3), 4);
  }

  // <int32_t, false>
  {
    ContiguousSpan<int32_t> cs_int = data22_int.flat();
    ASSERT_EQ(cs_int.size(), 4);
    ASSERT_FLOAT_EQ(cs_int.at(3), 4);
  }

  Array<float> lv = std::move(data14);
  ASSERT_FLOAT_EQ(lv.at(0, 0), 1);
  ASSERT_FLOAT_EQ(lv.at(0, 1), 2);
  ASSERT_FLOAT_EQ(lv.at(0, 2), 3);
  ASSERT_FLOAT_EQ(lv.at(0, 3), 4);
}

TEST(NDArray_tests, slice_write_test)
{
  // float
  {
    float raw_data[4] = {0};

    Array<float> data22{raw_data, {2, 2}};

    data22.slice(1) = {1, 2};

    ASSERT_FLOAT_EQ(data22.at(0, 0), 0);
    ASSERT_FLOAT_EQ(data22.at(0, 1), 0);
    ASSERT_FLOAT_EQ(data22.at(1, 0), 1);
    ASSERT_FLOAT_EQ(data22.at(1, 1), 2);
  }

  // int32_t
  {
    int32_t raw_data_int[4] = {0};
    Array<int32_t> data22{raw_data_int, {2, 2}};

    data22.slice(1) = {1, 2};

    ASSERT_FLOAT_EQ(data22.at(0, 0), 0);
    ASSERT_FLOAT_EQ(data22.at(0, 1), 0);
    ASSERT_FLOAT_EQ(data22.at(1, 0), 1);
    ASSERT_FLOAT_EQ(data22.at(1, 1), 2);
  }
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
  // float
  {
    std::vector<float> v1{1, 2, 3, 4, 5};
    std::vector<float> v2(5);

    ContiguousSpan<float> span1(v1.begin(), v1.end());
    ContiguousSpan<float> span2(v2.begin(), v2.end());

    span2.assign(span1);

    ASSERT_EQ(v1, v2);
    ASSERT_EQ(span1.size(), 5);
    ASSERT_EQ(span2.size(), 5);

    ASSERT_EQ(span2.at(2), 3);
    ASSERT_EQ(span2.at(4), 5);

    ASSERT_EQ(*(span1.data() + 2), *(span1.data() + 2));

    ContiguousSpan<float> span3(span2.offset(1));
    ASSERT_EQ(span3.size(), 4);
    ASSERT_EQ(span3.at(0), 2);
    ASSERT_EQ(span3.at(1), 3);
    ASSERT_EQ(span3.at(2), 4);
    ASSERT_EQ(span3.at(3), 5);

    // const
    ContiguousSpan<float, true> span3_const(span2.offset(1));
    ASSERT_EQ(span3_const.size(), 4);
    ASSERT_EQ(span3_const.at(0), 2);
    ASSERT_EQ(span3_const.at(1), 3);
    ASSERT_EQ(span3_const.at(2), 4);
    ASSERT_EQ(span3_const.at(3), 5);
  }

  // int32_t
  {
    std::vector<int32_t> v4{1, 2, 3, 4, 5};
    std::vector<int32_t> v5(5);

    ContiguousSpan<int32_t> span4(v4.begin(), v4.end());
    ContiguousSpan<int32_t> span5(v5.begin(), v5.end());

    span5.assign(span4);

    ASSERT_EQ(v4, v5);
    ASSERT_EQ(span4.size(), 5);
    ASSERT_EQ(span5.size(), 5);

    ASSERT_EQ(span5.at(2), 3);
    ASSERT_EQ(span5.at(4), 5);

    ASSERT_EQ(*(span4.data() + 2), *(span5.data() + 2));

    ContiguousSpan<int32_t> span6(span5.offset(1));
    ASSERT_EQ(span6.size(), 4);
    ASSERT_EQ(span6.at(0), 2);
    ASSERT_EQ(span6.at(1), 3);
    ASSERT_EQ(span6.at(2), 4);
    ASSERT_EQ(span6.at(3), 5);
  }
}
