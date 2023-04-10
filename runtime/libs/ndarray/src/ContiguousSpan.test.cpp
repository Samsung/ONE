/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ndarray/ContiguousSpan.h"

#include <gtest/gtest.h>

using namespace ndarray;

TEST(NDArrayContiguousSpanTests, slice_assign_test)
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

    ASSERT_EQ(*(span1.data() + 2), *(span2.data() + 2));

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

    ASSERT_EQ(*(span1.data() + 2), *(span2.data() + 2));

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

    ASSERT_EQ(*(span1.data() + 2), *(span2.data() + 2));

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

    ASSERT_EQ(*(span1.data() + 2), *(span2.data() + 2));

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
