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
  uint32_t raw_data_uint[] = {1, 2, 3, 4};
  int8_t raw_data_int8[] = {1, 2, 3, 4};

  Array<float> data22{raw_data, {2, 2}};
  Array<int32_t> data22_int{raw_data_int, {2, 2}};
  Array<uint32_t> data22_uint{raw_data_uint, {2, 2}};
  Array<int8_t> data22_int8{raw_data_int8, {2, 2}};

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

    ContiguousSpan<float> cs2 = std::move(cs);
    ASSERT_EQ(cs2.size(), 4);
    ASSERT_FLOAT_EQ(cs2.at(3), 4);

    float sum = 0;
    for (auto it = cs2.begin(); it < cs2.end(); it++)
    {
      sum += *it;
    }
    ASSERT_EQ(sum, 10);

    std::vector<float> array_data{1, 2, 3, 4};
    auto cs3 = std::make_unique<ContiguousSpan<float>>(array_data.begin(), array_data.end());
    ASSERT_EQ(cs3->size(), 4);
    ASSERT_FLOAT_EQ(cs3->at(3), 4);

    auto cs4 = std::move(cs3);
    ASSERT_EQ(cs3, nullptr);
    ASSERT_EQ(cs4->size(), 4);
    ASSERT_FLOAT_EQ(cs4->at(3), 4);
  }

  // <float, true>
  {
    ContiguousSpan<float, true> cs = data22.flat();
    ASSERT_EQ(cs.size(), 4);
    ASSERT_FLOAT_EQ(cs.at(3), 4);

    ContiguousSpan<float, true> cs2 = std::move(cs);
    ASSERT_EQ(cs2.size(), 4);
    ASSERT_FLOAT_EQ(cs2.at(3), 4);

    float sum = 0;
    for (auto it = cs2.begin(); it < cs2.end(); it++)
    {
      sum += *it;
    }
    ASSERT_FLOAT_EQ(sum, 10);

    std::vector<float> array_data{1, 2, 3, 4};
    auto cs3 = std::make_unique<ContiguousSpan<float, true>>(array_data.begin(), array_data.end());
    ASSERT_EQ(cs3->size(), 4);
    ASSERT_FLOAT_EQ(cs3->at(3), 4);

    auto cs4 = std::move(cs3);
    ASSERT_EQ(cs3, nullptr);
    ASSERT_EQ(cs4->size(), 4);
    ASSERT_FLOAT_EQ(cs4->at(3), 4);
  }

  // <int32_t, false>
  {
    ContiguousSpan<int32_t> cs = data22_int.flat();
    ASSERT_EQ(cs.size(), 4);
    ASSERT_EQ(cs.at(3), 4);

    ContiguousSpan<int32_t> cs2 = std::move(cs);
    ASSERT_EQ(cs2.size(), 4);
    ASSERT_EQ(cs2.at(3), 4);

    int32_t sum = 0;
    for (auto it = cs2.begin(); it < cs2.end(); it++)
    {
      sum += *it;
    }
    ASSERT_EQ(sum, 10);

    std::vector<int32_t> array_data{1, 2, 3, 4};
    auto cs3 = std::make_unique<ContiguousSpan<int32_t>>(array_data.begin(), array_data.end());
    ASSERT_EQ(cs3->size(), 4);
    ASSERT_EQ(cs3->at(3), 4);

    auto cs4 = std::move(cs3);
    ASSERT_EQ(cs3, nullptr);
    ASSERT_EQ(cs4->size(), 4);
    ASSERT_EQ(cs4->at(3), 4);
  }

  // <int32_t, true>
  {
    ContiguousSpan<int32_t, true> cs = data22_int.flat();
    ASSERT_EQ(cs.size(), 4);
    ASSERT_EQ(cs.at(3), 4);

    ContiguousSpan<int32_t, true> cs2 = std::move(cs);
    ASSERT_EQ(cs2.size(), 4);
    ASSERT_EQ(cs2.at(3), 4);

    int32_t sum = 0;
    for (auto it = cs2.begin(); it < cs2.end(); it++)
    {
      sum += *it;
    }
    ASSERT_EQ(sum, 10);

    std::vector<int32_t> array_data{1, 2, 3, 4};
    auto cs3 =
      std::make_unique<ContiguousSpan<int32_t, true>>(array_data.begin(), array_data.end());
    ASSERT_EQ(cs3->size(), 4);
    ASSERT_EQ(cs3->at(3), 4);

    auto cs4 = std::move(cs3);
    ASSERT_EQ(cs3, nullptr);
    ASSERT_EQ(cs4->size(), 4);
    ASSERT_EQ(cs4->at(3), 4);
  }

  // <uint32_t, false>
  {
    ContiguousSpan<uint32_t> cs = data22_uint.flat();
    ASSERT_EQ(cs.size(), 4);
    ASSERT_EQ(cs.at(3), 4);

    ContiguousSpan<uint32_t> cs2 = std::move(cs);
    ASSERT_EQ(cs2.size(), 4);
    ASSERT_EQ(cs2.at(3), 4);

    uint32_t sum = 0;
    for (auto it = cs2.begin(); it < cs2.end(); it++)
    {
      sum += *it;
    }
    ASSERT_EQ(sum, 10);

    std::vector<uint32_t> array_data{1, 2, 3, 4};
    auto cs3 = std::make_unique<ContiguousSpan<uint32_t>>(array_data.begin(), array_data.end());
    ASSERT_EQ(cs3->size(), 4);
    ASSERT_EQ(cs3->at(3), 4);

    auto cs4 = std::move(cs3);
    ASSERT_EQ(cs3, nullptr);
    ASSERT_EQ(cs4->size(), 4);
  }

  // <uint32_t, true>
  {
    ContiguousSpan<uint32_t, true> cs = data22_uint.flat();
    ASSERT_EQ(cs.size(), 4);
    ASSERT_EQ(cs.at(3), 4);

    ContiguousSpan<uint32_t, true> cs2 = std::move(cs);
    ASSERT_EQ(cs2.size(), 4);
    ASSERT_EQ(cs2.at(3), 4);

    uint32_t sum = 0;
    for (auto it = cs2.begin(); it < cs2.end(); it++)
    {
      sum += *it;
    }
    ASSERT_EQ(sum, 10);

    std::vector<uint32_t> array_data{1, 2, 3, 4};
    auto cs3 =
      std::make_unique<ContiguousSpan<uint32_t, true>>(array_data.begin(), array_data.end());
    ASSERT_EQ(cs3->size(), 4);
    ASSERT_EQ(cs3->at(3), 4);

    auto cs4 = std::move(cs3);
    ASSERT_EQ(cs3, nullptr);
    ASSERT_EQ(cs4->size(), 4);
    ASSERT_EQ(cs4->at(3), 4);
  }

  // <int8_t, false>
  {
    ContiguousSpan<int8_t> cs = data22_int8.flat();
    ASSERT_EQ(cs.size(), 4);
    ASSERT_FLOAT_EQ(cs.at(3), 4);

    ContiguousSpan<int8_t> cs2 = std::move(cs);
    ASSERT_EQ(cs2.size(), 4);
    ASSERT_FLOAT_EQ(cs2.at(3), 4);

    int8_t sum = 0;
    for (auto it = cs2.begin(); it < cs2.end(); it++)
    {
      sum += *it;
    }
    ASSERT_EQ(sum, 10);

    std::vector<int8_t> array_data{1, 2, 3, 4};
    auto cs3 = std::make_unique<ContiguousSpan<int8_t>>(array_data.begin(), array_data.end());
    ASSERT_EQ(cs3->size(), 4);
    ASSERT_EQ(cs3->at(3), 4);

    auto cs4 = std::move(cs3);
    ASSERT_EQ(cs3, nullptr);
    ASSERT_EQ(cs4->size(), 4);
    ASSERT_EQ(cs4->at(3), 4);

    auto cs5 = ContiguousSpan<int8_t>(array_data.begin(), array_data.end());
    ASSERT_EQ(cs5.size(), 4);
    ASSERT_EQ(cs5.at(3), 4);
  }

  // <int8_t, true>
  {
    ContiguousSpan<int8_t, true> cs = data22_int8.flat();
    ASSERT_EQ(cs.size(), 4);
    ASSERT_FLOAT_EQ(cs.at(3), 4);

    ContiguousSpan<int8_t, true> cs2 = std::move(cs);
    ASSERT_EQ(cs2.size(), 4);
    ASSERT_FLOAT_EQ(cs2.at(3), 4);

    int8_t sum = 0;
    for (auto it = cs2.begin(); it < cs2.end(); it++)
    {
      sum += *it;
    }
    ASSERT_EQ(sum, 10);

    std::vector<int8_t> array_data{1, 2, 3, 4};
    auto cs3 = std::make_unique<ContiguousSpan<int8_t, true>>(array_data.begin(), array_data.end());
    ASSERT_EQ(cs3->size(), 4);
    ASSERT_EQ(cs3->at(3), 4);

    auto cs4 = std::move(cs3);
    ASSERT_EQ(cs3, nullptr);
    ASSERT_EQ(cs4->size(), 4);
    ASSERT_EQ(cs4->at(3), 4);

    auto cs5 = ContiguousSpan<int8_t, true>(array_data.begin(), array_data.end());
    ASSERT_EQ(cs5.size(), 4);
    ASSERT_EQ(cs5.at(3), 4);
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
    int32_t raw_data[4] = {0};
    Array<int32_t> data22{raw_data, {2, 2}};

    data22.slice(1) = {1, 2};

    ASSERT_EQ(data22.at(0, 0), 0);
    ASSERT_EQ(data22.at(0, 1), 0);
    ASSERT_EQ(data22.at(1, 0), 1);
    ASSERT_EQ(data22.at(1, 1), 2);
  }

  // uint32_t
  {
    uint32_t raw_data[4] = {0};
    Array<uint32_t> data22{raw_data, {2, 2}};

    data22.slice(1) = {1, 2};

    ASSERT_EQ(data22.at(0, 0), 0);
    ASSERT_EQ(data22.at(0, 1), 0);
    ASSERT_EQ(data22.at(1, 0), 1);
    ASSERT_EQ(data22.at(1, 1), 2);
  }

  // int8_t
  {
    int8_t raw_data[4] = {0};
    Array<int8_t> data22{raw_data, {2, 2}};

    data22.slice(1) = {1, 2};

    ASSERT_EQ(data22.at(0, 0), 0);
    ASSERT_EQ(data22.at(0, 1), 0);
    ASSERT_EQ(data22.at(1, 0), 1);
    ASSERT_EQ(data22.at(1, 1), 2);
  }
}

TEST(NDArray_tests, slice_read_test)
{
  // float
  {
    float raw_data[4] = {1, 2, 3, 4};

    Array<float> data22{raw_data, {2, 2}};

    auto slice = data22.slice(1);

    ASSERT_FLOAT_EQ(slice[0], 3);
    ASSERT_FLOAT_EQ(slice[1], 4);
  }

  // int32_t
  {
    int32_t raw_data[4] = {1, 2, 3, 4};

    Array<int32_t> data22{raw_data, {2, 2}};

    auto slice = data22.slice(1);

    ASSERT_EQ(slice[0], 3);
    ASSERT_EQ(slice[1], 4);
  }

  // uint32_t
  {
    uint32_t raw_data[4] = {1, 2, 3, 4};

    Array<uint32_t> data22{raw_data, {2, 2}};

    auto slice = data22.slice(1);

    ASSERT_EQ(slice[0], 3);
    ASSERT_EQ(slice[1], 4);
  }

  // int8_t
  {
    int8_t raw_data[4] = {1, 2, 3, 4};

    Array<int8_t> data22{raw_data, {2, 2}};

    auto slice = data22.slice(1);

    ASSERT_EQ(slice[0], 3);
    ASSERT_EQ(slice[1], 4);
  }
}

TEST(NDArray_tests, multidim_test)
{
  // float
  {
    float raw_data[5] = {0, 1, 2, 3, 4};

    Array<float> data22{raw_data, {1, 1, 1, 1, 5}};

    ASSERT_FLOAT_EQ(data22.at(0, 0, 0, 0, 0), 0);
    ASSERT_FLOAT_EQ(data22.at(0, 0, 0, 0, 1), 1);
    ASSERT_FLOAT_EQ(data22.at(0, 0, 0, 0, 2), 2);
    ASSERT_FLOAT_EQ(data22.at(0, 0, 0, 0, 3), 3);
    ASSERT_FLOAT_EQ(data22.at(0, 0, 0, 0, 4), 4);
  }

  // int32_t
  {
    int32_t raw_data[5] = {0, 1, 2, 3, 4};

    Array<int32_t> data22{raw_data, {1, 1, 1, 1, 5}};

    ASSERT_EQ(data22.at(0, 0, 0, 0, 0), 0);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 1), 1);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 2), 2);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 3), 3);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 4), 4);
  }

  // uint32_t
  {
    uint32_t raw_data[5] = {0, 1, 2, 3, 4};

    Array<uint32_t> data22{raw_data, {1, 1, 1, 1, 5}};

    ASSERT_EQ(data22.at(0, 0, 0, 0, 0), 0);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 1), 1);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 2), 2);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 3), 3);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 4), 4);
  }

  // int8_t
  {
    int8_t raw_data[5] = {0, 1, 2, 3, 4};

    Array<int8_t> data22{raw_data, {1, 1, 1, 1, 5}};

    ASSERT_EQ(data22.at(0, 0, 0, 0, 0), 0);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 1), 1);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 2), 2);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 3), 3);
    ASSERT_EQ(data22.at(0, 0, 0, 0, 4), 4);
  }
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
    ASSERT_EQ(span3[2], 4);
    ASSERT_EQ(span3[3], 5);

    // const
    ContiguousSpan<float, true> span4(v1.begin(), v1.end());
    ASSERT_EQ(span4.size(), 5);
    ASSERT_EQ(span4.at(0), 1);
    ASSERT_EQ(span4.at(1), 2);
    ASSERT_EQ(span4.at(2), 3);
    ASSERT_EQ(span4[3], 4);
    ASSERT_EQ(span4[4], 5);

    ContiguousSpan<float, true> span5(span4.offset(1));
    ASSERT_EQ(span5.size(), 4);
    ASSERT_EQ(span5.at(0), 2);
    ASSERT_EQ(span5.at(1), 3);
    ASSERT_EQ(span5[2], 4);
    ASSERT_EQ(span5[3], 5);

    ASSERT_EQ(*(span5.data() + 2), *(span5.data() + 2));
  }

  // int32_t
  {
    std::vector<int32_t> v1{1, 2, 3, 4, 5};
    std::vector<int32_t> v2(5);

    ContiguousSpan<int32_t> span1(v1.begin(), v1.end());
    ContiguousSpan<int32_t> span2(v2.begin(), v2.end());

    span2.assign(span1);

    ASSERT_EQ(v1, v2);
    ASSERT_EQ(span1.size(), 5);
    ASSERT_EQ(span2.size(), 5);

    ASSERT_EQ(span2.at(2), 3);
    ASSERT_EQ(span2.at(4), 5);

    ASSERT_EQ(*(span1.data() + 2), *(span1.data() + 2));

    ContiguousSpan<int32_t> span3(span2.offset(1));
    ASSERT_EQ(span3.size(), 4);
    ASSERT_EQ(span3.at(0), 2);
    ASSERT_EQ(span3.at(1), 3);
    ASSERT_EQ(span3[2], 4);
    ASSERT_EQ(span3[3], 5);

    // const
    ContiguousSpan<int32_t, true> span4(v1.begin(), v1.end());
    ASSERT_EQ(span4.size(), 5);
    ASSERT_EQ(span4.at(0), 1);
    ASSERT_EQ(span4.at(1), 2);
    ASSERT_EQ(span4.at(2), 3);
    ASSERT_EQ(span4[3], 4);
    ASSERT_EQ(span4[4], 5);

    ContiguousSpan<int32_t, true> span5(span4.offset(1));
    ASSERT_EQ(span5.size(), 4);
    ASSERT_EQ(span5.at(0), 2);
    ASSERT_EQ(span5.at(1), 3);
    ASSERT_EQ(span5[2], 4);
    ASSERT_EQ(span5[3], 5);
  }

  // uint32_t
  {
    std::vector<uint32_t> v1{1, 2, 3, 4, 5};
    std::vector<uint32_t> v2(5);

    ContiguousSpan<uint32_t> span1(v1.begin(), v1.end());
    ContiguousSpan<uint32_t> span2(v2.begin(), v2.end());

    span2.assign(span1);

    ASSERT_EQ(v1, v2);
    ASSERT_EQ(span1.size(), 5);
    ASSERT_EQ(span2.size(), 5);

    ASSERT_EQ(span2.at(2), 3);
    ASSERT_EQ(span2.at(4), 5);

    ASSERT_EQ(*(span1.data() + 2), *(span1.data() + 2));

    ContiguousSpan<uint32_t> span3(span2.offset(1));
    ASSERT_EQ(span3.size(), 4);
    ASSERT_EQ(span3.at(0), 2);
    ASSERT_EQ(span3.at(1), 3);
    ASSERT_EQ(span3[2], 4);
    ASSERT_EQ(span3[3], 5);

    // const
    ContiguousSpan<uint32_t, true> span4(v1.begin(), v1.end());
    ASSERT_EQ(span4.size(), 5);
    ASSERT_EQ(span4.at(0), 1);
    ASSERT_EQ(span4.at(1), 2);
    ASSERT_EQ(span4.at(2), 3);
    ASSERT_EQ(span4[3], 4);
    ASSERT_EQ(span4[4], 5);

    ContiguousSpan<uint32_t, true> span5(span4.offset(1));
    ASSERT_EQ(span5.size(), 4);
    ASSERT_EQ(span5.at(0), 2);
    ASSERT_EQ(span5.at(1), 3);
    ASSERT_EQ(span5[2], 4);
    ASSERT_EQ(span5[3], 5);
  }

  // int8_t
  {
    std::vector<int8_t> v1{1, 2, 3, 4, 5};
    std::vector<int8_t> v2(5);

    ContiguousSpan<int8_t> span1(v1.begin(), v1.end());
    ContiguousSpan<int8_t> span2(v2.begin(), v2.end());

    span2.assign(span1);

    ASSERT_EQ(v1, v2);
    ASSERT_EQ(span1.size(), 5);
    ASSERT_EQ(span2.size(), 5);

    ASSERT_EQ(span2.at(2), 3);
    ASSERT_EQ(span2.at(4), 5);

    ASSERT_EQ(*(span1.data() + 2), *(span1.data() + 2));

    ContiguousSpan<int8_t> span3(span2.offset(1));
    ASSERT_EQ(span3.size(), 4);
    ASSERT_EQ(span3.at(0), 2);
    ASSERT_EQ(span3.at(1), 3);
    ASSERT_EQ(span3[2], 4);
    ASSERT_EQ(span3[3], 5);

    // const
    ContiguousSpan<int8_t, true> span4(v1.begin(), v1.end());
    ASSERT_EQ(span4.size(), 5);
    ASSERT_EQ(span4.at(0), 1);
    ASSERT_EQ(span4.at(1), 2);
    ASSERT_EQ(span4.at(2), 3);
    ASSERT_EQ(span4[3], 4);
    ASSERT_EQ(span4[4], 5);

    ContiguousSpan<int8_t, true> span5(span4.offset(1));
    ASSERT_EQ(span5.size(), 4);
    ASSERT_EQ(span5.at(0), 2);
    ASSERT_EQ(span5.at(1), 3);
    ASSERT_EQ(span5[2], 4);
    ASSERT_EQ(span5[3], 5);
  }
}
