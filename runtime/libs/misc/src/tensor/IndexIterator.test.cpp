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

#include "misc/tensor/IndexIterator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>

using namespace nnfw::misc::tensor;

TEST(MiscIndexIteratorTest, iterate)
{
  const Shape shape{3, 4, 7};

  std::array<int, 3 * 4 * 7> array;

  array.fill(0);

  iterate(shape) << [&](const Index &index) {
    assert(index.rank() == shape.rank());

    const uint32_t rank = index.rank();

    uint32_t offset = index.at(0);

    for (uint32_t axis = 1; axis < rank; ++axis)
    {
      offset *= shape.dim(axis);
      offset += index.at(axis);
    }

    array[offset] += 1;
  };

  ASSERT_TRUE(std::all_of(array.begin(), array.end(), [](int num) { return num == 1; }));
}

TEST(MiscIndexIteratorTest, neg_zero_rank_shape)
{
  // Test abnormal case of empty shape
  // It is expected not to throw any exception, do nothing
  const Shape shape{};

  ASSERT_NO_THROW(iterate(shape) << ([](const Index &) {}));
  SUCCEED();
}
