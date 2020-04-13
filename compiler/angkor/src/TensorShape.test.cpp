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

#include "angkor/TensorShape.h"

#include <gtest/gtest.h>

TEST(TensorShapeTest, ctor)
{
  angkor::TensorShape shape;

  ASSERT_EQ(shape.rank(), 0);
}

TEST(TensorShapeTest, ctor_initializer_list)
{
  angkor::TensorShape shape{1, 3, 5, 7};

  ASSERT_EQ(shape.rank(), 4);

  ASSERT_EQ(shape.dim(0), 1);
  ASSERT_EQ(shape.dim(1), 3);
  ASSERT_EQ(shape.dim(2), 5);
  ASSERT_EQ(shape.dim(3), 7);
}

TEST(TensorShapeTest, resize)
{
  angkor::TensorShape shape;

  shape.resize(4);

  ASSERT_EQ(shape.rank(), 4);
}

TEST(TensorShapeTest, dim)
{
  angkor::TensorShape shape;

  shape.resize(4);

  uint32_t dims[4] = {3, 5, 2, 7};

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    shape.dim(axis) = dims[axis];
    ASSERT_EQ(shape.dim(axis), dims[axis]);
  }
}

TEST(TensorShapeTest, copy)
{
  const angkor::TensorShape original{3, 5, 2, 7};
  const angkor::TensorShape copied{original};

  ASSERT_EQ(original.rank(), copied.rank());

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    ASSERT_EQ(original.dim(axis), copied.dim(axis));
  }
}

TEST(TensorShapeTest, eq_negative_on_unmatched_rank)
{
  const angkor::TensorShape left{1, 1, 1};
  const angkor::TensorShape right{1, 1, 1, 1};

  ASSERT_FALSE(left == right);
}

TEST(TensorShapeTest, eq_negative_on_unmatched_dim)
{
  const angkor::TensorShape left{2, 3};
  const angkor::TensorShape right{2, 4};

  ASSERT_FALSE(left == right);
}

TEST(TensorShapeTest, eq_positive)
{
  const angkor::TensorShape left{2, 3};
  const angkor::TensorShape right{2, 3};

  ASSERT_TRUE(left == right);
}
