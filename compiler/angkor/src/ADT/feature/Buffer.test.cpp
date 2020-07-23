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

#include "nncc/core/ADT/feature/Buffer.h"
#include "nncc/core/ADT/feature/CHWLayout.h"

#include <gtest/gtest.h>

using nncc::core::ADT::feature::Buffer;
using nncc::core::ADT::feature::CHWLayout;
using nncc::core::ADT::feature::Shape;

using nncc::core::ADT::feature::make_buffer;

TEST(ADT_FEATURE_BUFFER, ctor)
{
  const Shape shape{4, 6, 3};
  auto buffer = make_buffer<int, CHWLayout>(shape);

  ASSERT_EQ(shape.depth(), buffer.shape().depth());
  ASSERT_EQ(shape.height(), buffer.shape().height());
  ASSERT_EQ(shape.width(), buffer.shape().width());
}

TEST(ADT_FEATURE_BUFFER, access)
{
  const Shape shape{4, 6, 3};
  auto buffer = make_buffer<int, CHWLayout>(shape);

  ASSERT_EQ(0, buffer.at(3, 5, 2));
  buffer.at(3, 5, 2) = 4;

  // Casting is introduced to use 'const T &at(...) const' method
  ASSERT_EQ(4, static_cast<const Buffer<int> &>(buffer).at(3, 5, 2));
}
