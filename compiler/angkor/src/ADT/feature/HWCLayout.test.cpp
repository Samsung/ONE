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

#include "nncc/core/ADT/feature/HWCLayout.h"

#include <gtest/gtest.h>

using namespace nncc::core::ADT::feature;

TEST(ADT_FEATURE_HWC_LAYOUT, C_increase)
{
  const uint32_t C = 4;
  const uint32_t H = 3;
  const uint32_t W = 6;

  const Shape shape{C, H, W};
  const HWCLayout l;

  ASSERT_EQ(l.offset(shape, 2, 1, 1), l.offset(shape, 1, 1, 1) + 1);
}

TEST(ADT_FEATURE_HWC_LAYOUT, W_increase)
{
  const uint32_t C = 4;
  const uint32_t H = 3;
  const uint32_t W = 6;

  const Shape shape{C, H, W};
  const HWCLayout l;

  ASSERT_EQ(l.offset(shape, 1, 2, 2), l.offset(shape, 1, 2, 1) + C);
}

TEST(ADT_FEATURE_HWC_LAYOUT, H_increase)
{
  const uint32_t C = 4;
  const uint32_t H = 3;
  const uint32_t W = 6;

  const Shape shape{C, H, W};
  const HWCLayout l;

  ASSERT_EQ(l.offset(shape, 1, 2, 1), l.offset(shape, 1, 1, 1) + W * C);
}
