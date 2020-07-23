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

#include "nncc/core/ADT/kernel/NHWCLayout.h"

#include <gtest/gtest.h>

using nncc::core::ADT::kernel::NHWCLayout;
using nncc::core::ADT::kernel::Shape;

TEST(ADT_KERNEL_KERNEL_NHWC_LAYOUT, ch_increment)
{
  const uint32_t N = 4;
  const uint32_t C = 3;
  const uint32_t H = 6;
  const uint32_t W = 5;

  const Shape shape{N, C, H, W};
  const NHWCLayout l;

  ASSERT_EQ(l.offset(shape, 1, 2, 1, 1), l.offset(shape, 1, 1, 1, 1) + 1);
}

TEST(ADT_KERNEL_KERNEL_NHWC_LAYOUT, col_increment)
{
  const uint32_t N = 4;
  const uint32_t C = 3;
  const uint32_t H = 6;
  const uint32_t W = 5;

  const Shape shape{N, C, H, W};
  const NHWCLayout l;

  ASSERT_EQ(l.offset(shape, 1, 1, 1, 2), l.offset(shape, 1, 1, 1, 1) + C);
}

TEST(ADT_KERNEL_KERNEL_NHWC_LAYOUT, row_increment)
{
  const uint32_t N = 4;
  const uint32_t C = 3;
  const uint32_t H = 6;
  const uint32_t W = 5;

  const Shape shape{N, C, H, W};
  const NHWCLayout l;

  ASSERT_EQ(l.offset(shape, 1, 1, 2, 1), l.offset(shape, 1, 1, 1, 1) + C * W);
}

TEST(ADT_KERNEL_KERNEL_NHWC_LAYOUT, n_increment)
{
  const uint32_t N = 4;
  const uint32_t C = 3;
  const uint32_t H = 6;
  const uint32_t W = 5;

  const Shape shape{N, C, H, W};
  const NHWCLayout l;

  ASSERT_EQ(l.offset(shape, 2, 1, 1, 1), l.offset(shape, 1, 1, 1, 1) + H * W * C);
}
