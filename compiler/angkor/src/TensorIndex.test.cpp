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

  ASSERT_EQ(0, index.rank());
}

TEST(TensorIndexTest, ctor_initializer_list)
{
  const angkor::TensorIndex index{1, 3, 5, 7};

  ASSERT_EQ(4, index.rank());

  ASSERT_EQ(1, index.at(0));
  ASSERT_EQ(3, index.at(1));
  ASSERT_EQ(5, index.at(2));
  ASSERT_EQ(7, index.at(3));
}

TEST(TensorIndexTest, resize)
{
  angkor::TensorIndex index;

  index.resize(4);

  ASSERT_EQ(4, index.rank());
}

TEST(TensorIndexTest, at)
{
  angkor::TensorIndex index;

  index.resize(4);

  uint32_t indices[4] = {3, 5, 2, 7};

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    index.at(axis) = indices[axis];
    ASSERT_EQ(indices[axis], index.at(axis));
  }
}

TEST(TensorIndexTest, copy)
{
  const angkor::TensorIndex original{3, 5, 2, 7};
  const angkor::TensorIndex copied{original};

  ASSERT_EQ(copied.rank(), original.rank());

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    ASSERT_EQ(copied.at(axis), original.at(axis));
  }
}

TEST(TensorIndexTest, fill)
{
  angkor::TensorIndex index{1, 6};

  index.fill(3);

  ASSERT_EQ(2, index.rank());

  ASSERT_EQ(3, index.at(0));
  ASSERT_EQ(3, index.at(1));
}
