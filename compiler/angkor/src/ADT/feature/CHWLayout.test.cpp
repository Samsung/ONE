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

#include "nncc/core/ADT/feature/CHWLayout.h"

#include <gtest/gtest.h>

using namespace nncc::core::ADT::feature;

TEST(ADT_FEATURE_CHW_LAYOUT, col_increase)
{
  const Shape shape{4, 3, 6};
  const CHWLayout l;

  ASSERT_EQ(l.offset(shape, 1, 2, 2), l.offset(shape, 1, 2, 1) + 1);
}

TEST(ADT_FEATURE_CHW_LAYOUT, row_increase)
{
  const Shape shape{4, 3, 6};
  const CHWLayout l;

  ASSERT_EQ(l.offset(shape, 1, 2, 1), l.offset(shape, 1, 1, 1) + 6);
}

TEST(ADT_FEATURE_CHW_LAYOUT, ch_increase)
{
  const Shape shape{4, 3, 6};
  const CHWLayout l;

  ASSERT_EQ(l.offset(shape, 2, 1, 1), l.offset(shape, 1, 1, 1) + 6 * 3);
}
