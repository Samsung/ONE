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

#include "nncc/core/ADT/tensor/Shape.h"

#include <gtest/gtest.h>

TEST(ADT_TENSOR_SHAPE, ctor)
{
  nncc::core::ADT::tensor::Shape shape;

  ASSERT_EQ(0, shape.rank());
}

TEST(ADT_TENSOR_SHAPE, ctor_initializer_list)
{
  nncc::core::ADT::tensor::Shape shape{1, 3, 5, 7};

  ASSERT_EQ(4, shape.rank());

  ASSERT_EQ(1, shape.dim(0));
  ASSERT_EQ(3, shape.dim(1));
  ASSERT_EQ(5, shape.dim(2));
  ASSERT_EQ(7, shape.dim(3));
}

TEST(ADT_TENSOR_SHAPE, resize)
{
  nncc::core::ADT::tensor::Shape shape;

  shape.resize(4);

  ASSERT_EQ(4, shape.rank());
}

TEST(ADT_TENSOR_SHAPE, dim)
{
  nncc::core::ADT::tensor::Shape shape;

  shape.resize(4);

  uint32_t dims[4] = {3, 5, 2, 7};

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    shape.dim(axis) = dims[axis];
    ASSERT_EQ(dims[axis], shape.dim(axis));
  }
}

TEST(ADT_TENSOR_SHAPE, copy)
{
  const nncc::core::ADT::tensor::Shape original{3, 5, 2, 7};
  const nncc::core::ADT::tensor::Shape copied{original};

  ASSERT_EQ(copied.rank(), original.rank());

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    ASSERT_EQ(copied.dim(axis), original.dim(axis));
  }
}

TEST(ADT_TENSOR_SHAPE, num_elements_rank_0)
{
  using nncc::core::ADT::tensor::num_elements;
  using nncc::core::ADT::tensor::Shape;

  Shape rank_0_shape;

  ASSERT_EQ(1, num_elements(rank_0_shape));
}

TEST(ADT_TENSOR_SHAPE, num_elements_zero)
{
  using nncc::core::ADT::tensor::num_elements;
  using nncc::core::ADT::tensor::Shape;

  ASSERT_EQ(0, num_elements(Shape{0, 0, 0, 0}));
}

TEST(ADT_TENSOR_SHAPE, num_elements_nonzero)
{
  using nncc::core::ADT::tensor::num_elements;
  using nncc::core::ADT::tensor::Shape;

  ASSERT_EQ(6, num_elements(Shape{2, 3}));
}

TEST(ADT_TENSOR_SHAPE, num_elements_nulldim)
{
  using nncc::core::ADT::tensor::num_elements;
  using nncc::core::ADT::tensor::Shape;

  ASSERT_EQ(0, num_elements(Shape{2, 0, 3}));
}

TEST(ADT_TENSOR_SHAPE, squeeze_neg)
{
  using nncc::core::ADT::tensor::Shape;
  using nncc::core::ADT::tensor::squeeze;

  auto squeezed = squeeze(Shape{3, 5, 2});

  ASSERT_EQ(3, squeezed.rank());
  ASSERT_EQ(3, squeezed.dim(0));
  ASSERT_EQ(5, squeezed.dim(1));
  ASSERT_EQ(2, squeezed.dim(2));
}

TEST(ADT_TENSOR_SHAPE, squeeze_neg_0)
{
  using nncc::core::ADT::tensor::Shape;
  using nncc::core::ADT::tensor::squeeze;

  auto squeezed = squeeze(Shape{3, 0, 2});

  ASSERT_EQ(3, squeezed.rank());
  ASSERT_EQ(3, squeezed.dim(0));
  ASSERT_EQ(0, squeezed.dim(1));
  ASSERT_EQ(2, squeezed.dim(2));
}

TEST(ADT_TENSOR_SHAPE, squeeze_pos)
{
  using nncc::core::ADT::tensor::Shape;
  using nncc::core::ADT::tensor::squeeze;

  auto squeezed = squeeze(Shape{3, 1, 2});

  ASSERT_EQ(2, squeezed.rank());
  ASSERT_EQ(3, squeezed.dim(0));
  ASSERT_EQ(2, squeezed.dim(1));
}

TEST(ADT_TENSOR_SHAPE, squeeze_nested)
{
  using nncc::core::ADT::tensor::Shape;
  using nncc::core::ADT::tensor::squeeze;

  Shape shape{3, 1, 2};

  shape.squeeze().squeeze();

  ASSERT_EQ(2, shape.rank());
  ASSERT_EQ(3, shape.dim(0));
  ASSERT_EQ(2, shape.dim(1));
}

TEST(ADT_TENSOR_SHAPE, eq_negative_on_unmatched_rank)
{
  const nncc::core::ADT::tensor::Shape left{1, 1, 1};
  const nncc::core::ADT::tensor::Shape right{1, 1, 1, 1};

  ASSERT_FALSE(left == right);
}

TEST(ADT_TENSOR_SHAPE, eq_negative_on_unmatched_dim)
{
  const nncc::core::ADT::tensor::Shape left{2, 3};
  const nncc::core::ADT::tensor::Shape right{2, 4};

  ASSERT_FALSE(left == right);
}

TEST(ADT_TENSOR_SHAPE, eq_positive)
{
  const nncc::core::ADT::tensor::Shape left{2, 3};
  const nncc::core::ADT::tensor::Shape right{2, 3};

  ASSERT_TRUE(left == right);
}
