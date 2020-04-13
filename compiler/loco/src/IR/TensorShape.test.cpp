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

#include "loco/IR/TensorShape.h"

#include <gtest/gtest.h>

TEST(TensorShapeTest, default_constructor)
{
  loco::TensorShape tensor_shape;

  ASSERT_EQ(tensor_shape.rank(), 0);
}

TEST(TensorShapeTest, rank)
{
  loco::TensorShape tensor_shape;

  tensor_shape.rank(2);

  ASSERT_EQ(tensor_shape.rank(), 2);
  ASSERT_FALSE(tensor_shape.dim(0).known());
  ASSERT_FALSE(tensor_shape.dim(1).known());
}

TEST(TensorShapeTest, dim)
{
  loco::TensorShape tensor_shape;

  tensor_shape.rank(2);

  tensor_shape.dim(0) = 3;

  ASSERT_TRUE(tensor_shape.dim(0).known());
  ASSERT_FALSE(tensor_shape.dim(1).known());

  ASSERT_EQ(tensor_shape.dim(0), 3);
}

TEST(TensorShapeTest, rank_update)
{
  loco::TensorShape tensor_shape;

  tensor_shape.rank(2);

  tensor_shape.dim(1) = 3;

  tensor_shape.rank(4);

  ASSERT_FALSE(tensor_shape.dim(0).known());
  ASSERT_TRUE(tensor_shape.dim(1).known());
  ASSERT_FALSE(tensor_shape.dim(2).known());
  ASSERT_FALSE(tensor_shape.dim(3).known());

  ASSERT_EQ(tensor_shape.dim(1), 3);
}

TEST(TensorShapeTest, copy)
{
  loco::TensorShape src;

  src.rank(2);
  src.dim(1) = 3;

  loco::TensorShape dst;

  dst = src;

  ASSERT_EQ(dst.rank(), 2);

  ASSERT_FALSE(dst.dim(0).known());
  ASSERT_TRUE(dst.dim(1).known());

  ASSERT_EQ(dst.dim(1), 3);
}

TEST(TensorShapeTest, element_count)
{
  // Check Rank-0 case
  loco::TensorShape src;

  ASSERT_EQ(loco::element_count(&src), 1);
}
