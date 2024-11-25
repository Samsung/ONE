/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cker/ShapeIterator.h>
#include <cker/Utils.h>
#include <gtest/gtest.h>
#include <numeric>

using namespace nnfw::cker;

TEST(CKer_Utils, ShapeIterator_basic)
{
  const Shape test_shape{1, 3, 1024, 768};
  {
    // test the front and back iterability with basic operators
    ShapeIterator it{test_shape};
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(*it, 3);
    it++;
    EXPECT_EQ(*it, 1024);
    --it;
    EXPECT_EQ(*it, 3);
    it--;
    EXPECT_EQ(*it, 1);
  }
  {
    // test the iterator's compatibility with STL iterator functions
    ShapeIterator it{test_shape};
    auto it2 = std::next(it);
    EXPECT_EQ(*it2, 3);
    EXPECT_EQ(*it, 1); // make sure the original iterator is untouched

    std::advance(it2, 2);
    EXPECT_EQ(*it2, 768);

    std::advance(it2, -1);
    EXPECT_EQ(*it2, 1024);
  }
  {
    // postincrement operator test
    ShapeIterator it{test_shape};
    const auto it2 = it++;
    EXPECT_EQ(*it, 3);
    EXPECT_EQ(*it2, 1);
  }
  {
    // test the ability to iterate over a Shape with range-based loops
    int expected_dims[] = {1, 3, 1024, 768};
    int i = 0;
    for (auto &&dim : test_shape)
    {
      EXPECT_EQ(dim, expected_dims[i++]);
    }
  }
  {
    // test the ability to retrieve iterators using begin & end
    const auto first = begin(test_shape);
    const auto last = end(test_shape);
    EXPECT_GT(std::distance(first, last), 0);
    EXPECT_EQ(std::distance(first, last), test_shape.DimensionsCount());
  }

  {
    // test and demostrate the usage of iterators with STL algos
    const auto first = begin(test_shape);
    const auto last = end(test_shape);
    const auto shape_elems =
      std::accumulate(first, last, 1, std::multiplies<ShapeIterator::value_type>{});
    EXPECT_EQ(shape_elems, test_shape.FlatSize());
  }

  {
    // Shape and ofstream interoperability test
    std::stringstream ss;
    ss << test_shape;
    EXPECT_EQ(ss.str(), "[1,3,1024,768]");
  }
}

TEST(CKer_Utils, neg_ShapeIterator_empty_shape)
{
  const Shape test_shape{};
  {
    const auto first = begin(test_shape);
    const auto last = end(test_shape);
    EXPECT_EQ(first, last);
  }

  {
    std::stringstream ss;
    ss << test_shape;
    EXPECT_EQ(ss.str(), "[]");
  }
}
