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

#include "coco/IR/FeatureLayouts.h"

#include <gtest/gtest.h>

using namespace nncc::core::ADT;

TEST(FeatureLayoutsTest, BC)
{
  // NOTE The current implementation uses a hard-coded "batch" value
  const uint32_t B = 1;
  const uint32_t C = 3;
  const uint32_t H = 4;
  const uint32_t W = 5;

  auto l = coco::FeatureLayouts::BC::create(feature::Shape{C, H, W});

  ASSERT_EQ(l->batch(), B);
  ASSERT_EQ(l->depth(), C);
  ASSERT_EQ(l->height(), H);
  ASSERT_EQ(l->width(), W);

  // Check whether BC layout is actually channel-wise
  for (uint32_t b = 0; b < B; ++b)
  {
    for (uint32_t ch = 0; ch < C; ++ch)
    {
      for (uint32_t row = 0; row < H; ++row)
      {
        for (uint32_t col = 0; col < W; ++col)
        {
          ASSERT_EQ(l->at(b, ch, 0, 0), l->at(b, ch, row, col));
        }
      }
    }
  }

  // Check whether BC layout is actually channel-major
  for (uint32_t b = 0; b < B; ++b)
  {
    for (uint32_t ch = 1; ch < C; ++ch)
    {
      ASSERT_EQ(l->at(b, ch - 1, 0, 0).value() + 1, l->at(b, ch, 0, 0).value());
    }
  }

  for (uint32_t b = 1; b < B; ++b)
  {
    ASSERT_EQ(l->at(b - 1, C - 1, 0, 0).value() + 1, l->at(b, 0, 0, 0).value());
  }
}
