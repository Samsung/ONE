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

#include "coco/IR/KernelLayouts.h"

#include <nncc/core/ADT/kernel/NCHWLayout.h>
#include <nncc/core/ADT/kernel/NHWCLayout.h>

#include <gtest/gtest.h>

using namespace nncc::core::ADT;

TEST(KernelLayoutsTest, NCHW_increment)
{
  const uint32_t N = 2;
  const uint32_t C = 3;
  const uint32_t H = 4;
  const uint32_t W = 4;

  auto l = coco::KernelLayouts::NCHW::create(kernel::Shape{N, C, H, W});

  // check NCHW order
  ASSERT_EQ(l->at(0, 0, 0, 1).value(), l->at(0, 0, 0, 0).value() + 1);
  ASSERT_EQ(l->at(0, 0, 1, 0).value(), l->at(0, 0, 0, 0).value() + W);
  ASSERT_EQ(l->at(0, 1, 0, 0).value(), l->at(0, 0, 0, 0).value() + H * W);
  ASSERT_EQ(l->at(1, 0, 0, 0).value(), l->at(0, 0, 0, 0).value() + C * H * W);
}

TEST(KernelLayoutsTest, NHWC_increment)
{
  const uint32_t N = 2;
  const uint32_t C = 3;
  const uint32_t H = 4;
  const uint32_t W = 4;

  auto l = coco::KernelLayouts::NHWC::create(kernel::Shape{N, C, H, W});

  // check NHWC order
  ASSERT_EQ(l->at(0, 1, 0, 0).value(), l->at(0, 0, 0, 0).value() + 1);
  ASSERT_EQ(l->at(0, 0, 0, 1).value(), l->at(0, 0, 0, 0).value() + C);
  ASSERT_EQ(l->at(0, 0, 1, 0).value(), l->at(0, 0, 0, 0).value() + W * C);
  ASSERT_EQ(l->at(1, 0, 0, 0).value(), l->at(0, 0, 0, 0).value() + H * W * C);
}

TEST(KernelLayoutsTest, Generic_increment)
{
  const uint32_t N = 2;
  const uint32_t C = 3;
  const uint32_t H = 4;
  const uint32_t W = 4;

  auto nchw = coco::KernelLayouts::Generic::create(kernel::Shape{N, C, H, W});
  auto nhwc = coco::KernelLayouts::Generic::create(kernel::Shape{N, C, H, W});

  // reorder
  nchw->reorder(kernel::NCHWLayout());
  nhwc->reorder(kernel::NHWCLayout());

  // check NCHW order
  ASSERT_EQ(nchw->at(0, 0, 0, 1).value(), nchw->at(0, 0, 0, 0).value() + 1);
  ASSERT_EQ(nchw->at(0, 0, 1, 0).value(), nchw->at(0, 0, 0, 0).value() + W);
  ASSERT_EQ(nchw->at(0, 1, 0, 0).value(), nchw->at(0, 0, 0, 0).value() + H * W);
  ASSERT_EQ(nchw->at(1, 0, 0, 0).value(), nchw->at(0, 0, 0, 0).value() + C * H * W);

  // check NHWC order
  ASSERT_EQ(nhwc->at(0, 1, 0, 0).value(), nhwc->at(0, 0, 0, 0).value() + 1);
  ASSERT_EQ(nhwc->at(0, 0, 0, 1).value(), nhwc->at(0, 0, 0, 0).value() + C);
  ASSERT_EQ(nhwc->at(0, 0, 1, 0).value(), nhwc->at(0, 0, 0, 0).value() + W * C);
  ASSERT_EQ(nhwc->at(1, 0, 0, 0).value(), nhwc->at(0, 0, 0, 0).value() + H * W * C);
}

TEST(KernelLayoutsTest, Generic_at)
{
  const uint32_t N = 2;
  const uint32_t C = 3;
  const uint32_t H = 4;
  const uint32_t W = 4;

  auto l = coco::KernelLayouts::Generic::create(kernel::Shape{N, C, H, W});

  ASSERT_NE(l.get(), nullptr);

  coco::KernelLayouts::Generic *mutable_ptr = l.get();
  const coco::KernelLayouts::Generic *immutable_ptr = l.get();

  for (uint32_t n = 0; n < N; ++n)
  {
    for (uint32_t ch = 0; ch < C; ++ch)
    {
      for (uint32_t row = 0; row < H; ++row)
      {
        for (uint32_t col = 0; col < W; ++col)
        {
          mutable_ptr->at(n, ch, row, col) = coco::ElemID{16};
        }
      }
    }
  }

  for (uint32_t n = 0; n < N; ++n)
  {
    for (uint32_t ch = 0; ch < C; ++ch)
    {
      for (uint32_t row = 0; row < H; ++row)
      {
        for (uint32_t col = 0; col < W; ++col)
        {
          ASSERT_EQ(immutable_ptr->at(n, ch, row, col).value(), 16);
        }
      }
    }
  }
}
