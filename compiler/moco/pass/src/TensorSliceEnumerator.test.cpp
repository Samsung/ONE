/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TensorSliceEnumerator.h"

#include <gtest/gtest.h>

TEST(TensorSliceEnumeratorTest, basic_vector)
{
  moco::TensorSliceEnumerator iter;
  loco::TensorShape shape;
  uint32_t rank = 1;

  shape.rank(rank);
  shape.dim(0) = loco::Dimension(4);

  std::vector<uint32_t> begin = {1};
  std::vector<uint32_t> end = {3};

  iter.shape(shape);
  iter.begin(begin);
  iter.end(end);

  for (iter.start(); iter.valid(); iter.advance())
  {
    for (uint32_t r = 0; r < rank; ++r)
    {
      printf("%d ", iter.cursor(r));
    }
    printf("\n");
  }

  GTEST_SUCCEED();
}

TEST(TensorSliceEnumeratorTest, basic_matrix)
{
  moco::TensorSliceEnumerator etor;
  loco::TensorShape shape;
  uint32_t rank = 2;

  shape.rank(rank);
  shape.dim(0) = loco::Dimension(5);
  shape.dim(1) = loco::Dimension(5);

  std::vector<uint32_t> begin = {1, 1};
  std::vector<uint32_t> end = {2, 4};
  std::vector<uint32_t> offset;
  std::vector<uint32_t> cursor;

  etor.shape(shape);
  etor.begin(begin);
  etor.end(end);

  for (etor.start(); etor.valid(); etor.advance())
  {
    cursor = etor.cursor();
    assert(cursor.size() == begin.size());

    offset.resize(cursor.size());
    for (uint32_t r = 0; r < cursor.size(); r++)
    {
      offset.at(r) = cursor.at(r) - begin.at(r);
      std::cout << offset.at(r) << " ";
    }
    std::cout << std::endl;
  }

  GTEST_SUCCEED();
}
