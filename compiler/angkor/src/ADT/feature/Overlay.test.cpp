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

#include "nncc/core/ADT/feature/Overlay.h"
#include "nncc/core/ADT/feature/CHWLayout.h"

#include <gtest/gtest.h>

using nncc::core::ADT::feature::CHWLayout;
using nncc::core::ADT::feature::Overlay;
using nncc::core::ADT::feature::Shape;

using nncc::core::ADT::feature::make_overlay;

TEST(ADT_FEATURE_OVERLAY, ctor)
{
  const Shape shape{4, 6, 3};

  int data[4 * 6 * 3] = {
      0,
  };
  auto overlay = make_overlay<int, CHWLayout>(shape, data);

  ASSERT_EQ(shape.depth(), overlay.shape().depth());
  ASSERT_EQ(shape.height(), overlay.shape().height());
  ASSERT_EQ(shape.width(), overlay.shape().width());
}

TEST(ADT_FEATURE_OVERLAY, read)
{
  const Shape shape{4, 6, 3};

  int data[4 * 6 * 3] = {
      0,
  };
  const auto overlay = make_overlay<int, CHWLayout>(shape, data);

  CHWLayout layout{};

  ASSERT_EQ(0, data[layout.offset(shape, 3, 5, 2)]);
  data[layout.offset(shape, 3, 5, 2)] = 2;
  ASSERT_EQ(2, overlay.at(3, 5, 2));
}

TEST(ADT_FEATURE_OVERLAY, access)
{
  const Shape shape{4, 6, 3};

  int data[4 * 6 * 3] = {
      0,
  };
  auto overlay = make_overlay<int, CHWLayout>(shape, data);

  CHWLayout layout{};

  ASSERT_EQ(0, data[layout.offset(shape, 3, 5, 2)]);
  overlay.at(3, 5, 2) = 4;
  ASSERT_EQ(4, data[layout.offset(shape, 3, 5, 2)]);
}
