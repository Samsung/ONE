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

#include "nncc/core/ADT/feature/Layout.h"

#include <gtest/gtest.h>

using nncc::core::ADT::feature::Layout;
using nncc::core::ADT::feature::Shape;

static uint32_t offset_0(const Shape &, uint32_t, uint32_t, uint32_t) { return 0; }
static uint32_t offset_1(const Shape &, uint32_t, uint32_t, uint32_t) { return 1; }

TEST(ADT_FEATURE_LAYOUT, ctor)
{
  Layout l{offset_0};

  ASSERT_EQ(0, l.offset(Shape{4, 3, 6}, 1, 1, 1));
}

TEST(ADT_FEATURE_LAYOUT, copy)
{
  Layout orig{offset_0};
  Layout copy{offset_1};

  ASSERT_EQ(1, copy.offset(Shape{4, 3, 6}, 1, 1, 1));

  copy = orig;

  ASSERT_EQ(0, copy.offset(Shape{4, 3, 6}, 1, 1, 1));
}

TEST(ADT_FEATURE_LAYOUT, move)
{
  Layout orig{offset_0};
  Layout move{offset_1};

  ASSERT_EQ(1, move.offset(Shape{4, 3, 6}, 1, 1, 1));

  move = std::move(orig);

  ASSERT_EQ(0, move.offset(Shape{4, 3, 6}, 1, 1, 1));
}
