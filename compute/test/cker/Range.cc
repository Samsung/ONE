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

#include <cker/operation/Range.h>

#include <gtest/gtest.h>
#include <vector>

TEST(CKer_Operation, Range)
{
  {
    const int start = 0;
    const int limit = 10;
    const int delta = 1;
    std::vector<int> actual(10);
    nnfw::cker::Range<int>(&start, &limit, &delta, actual.data());

    for (int i = 0; i < actual.size(); i++)
      ASSERT_EQ(actual[i], i);
  }

  {
    const int start = 3;
    const int limit = 18;
    const int delta = 3;
    std::vector<int> expected = {3, 6, 9, 12, 15};
    std::vector<int> actual(expected.size());
    nnfw::cker::Range<int>(&start, &limit, &delta, actual.data());

    for (int i = 0; i < actual.size(); i++)
      ASSERT_EQ(actual[i], expected[i]);
  }

  {
    const float start = 3;
    const float limit = 1;
    const float delta = -0.5;
    std::vector<float> expected = {
        3,
        2.5,
        2,
        1.5,
    };
    std::vector<float> actual(expected.size());
    nnfw::cker::Range<float>(&start, &limit, &delta, actual.data());

    for (int i = 0; i < actual.size(); i++)
      ASSERT_FLOAT_EQ(actual[i], expected[i]);
  }
}

TEST(CKer_Operation, neg_Range)
{
  {
    const int start = 212;
    const int limit = 10;
    const int delta = 1;
    std::vector<int> actual(10);

    EXPECT_ANY_THROW(nnfw::cker::Range<int>(&start, &limit, &delta, actual.data()));
  }
}
