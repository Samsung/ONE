/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "angkor/TensorIndex.h"

#include <gtest/gtest.h>

TEST(TensorIndexTest, ctor)
{
  angkor::TensorIndex index;

  ASSERT_EQ(index.rank(), 0);
}

TEST(TensorIndexTest, ctor_initializer_list)
{
  const angkor::TensorIndex index{1, 3, 5, 7};

  ASSERT_EQ(index.rank(), 4);

  ASSERT_EQ(index.at(0), 1);
  ASSERT_EQ(index.at(1), 3);
  ASSERT_EQ(index.at(2), 5);
  ASSERT_EQ(index.at(3), 7);
}

TEST(TensorIndexTest, resize)
{
  angkor::TensorIndex index;

  index.resize(4);

  ASSERT_EQ(index.rank(), 4);
}

TEST(TensorIndexTest, at)
{
  angkor::TensorIndex index;

  index.resize(4);

  uint32_t indices[4] = {3, 5, 2, 7};

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    index.at(axis) = indices[axis];
    ASSERT_EQ(index.at(axis), indices[axis]);
  }
}

TEST(TensorIndexTest, copy)
{
  const angkor::TensorIndex original{3, 5, 2, 7};
  const angkor::TensorIndex copied{original};

  ASSERT_EQ(original.rank(), copied.rank());

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    ASSERT_EQ(original.at(axis), copied.at(axis));
  }
}

TEST(TensorIndexTest, fill)
{
  angkor::TensorIndex index{1, 6};

  index.fill(3);

  ASSERT_EQ(index.rank(), 2);

  ASSERT_EQ(index.at(0), 3);
  ASSERT_EQ(index.at(1), 3);
}
