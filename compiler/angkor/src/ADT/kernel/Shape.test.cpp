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

#include <nncc/core/ADT/kernel/Shape.h>

#include <gtest/gtest.h>

TEST(ADT_KERNEL_SHAPE, ctor)
{
  const uint32_t N = 1;
  const uint32_t C = 3;
  const uint32_t H = 4;
  const uint32_t W = 5;

  nncc::core::ADT::kernel::Shape shape{N, C, H, W};

  ASSERT_EQ(N, shape.count());
  ASSERT_EQ(C, shape.depth());
  ASSERT_EQ(H, shape.height());
  ASSERT_EQ(W, shape.width());
}

TEST(ADT_KERNEL_SHAPE, num_elements)
{
  const uint32_t N = 1;
  const uint32_t C = 3;
  const uint32_t H = 4;
  const uint32_t W = 5;

  using nncc::core::ADT::kernel::num_elements;
  using nncc::core::ADT::kernel::Shape;

  ASSERT_EQ(N * C * H * W, num_elements(Shape{N, C, H, W}));
}

TEST(ADT_KERNEL_SHAPE, operator_eq)
{
  using nncc::core::ADT::kernel::Shape;

  EXPECT_TRUE(Shape(1, 1, 1, 1) == Shape(1, 1, 1, 1));
  EXPECT_FALSE(Shape(1, 1, 1, 1) == Shape(1, 1, 1, 2));
  EXPECT_FALSE(Shape(1, 1, 1, 1) == Shape(1, 1, 2, 1));
  EXPECT_FALSE(Shape(1, 1, 1, 1) == Shape(1, 2, 1, 1));
  EXPECT_FALSE(Shape(1, 1, 1, 1) == Shape(2, 1, 1, 1));
}
