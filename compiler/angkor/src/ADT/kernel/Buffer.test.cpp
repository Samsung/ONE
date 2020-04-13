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

#include "nncc/core/ADT/kernel/Buffer.h"
#include "nncc/core/ADT/kernel/NCHWLayout.h"

#include <gtest/gtest.h>

using nncc::core::ADT::kernel::Shape;
using nncc::core::ADT::kernel::NCHWLayout;
using nncc::core::ADT::kernel::Buffer;

using nncc::core::ADT::kernel::make_buffer;

TEST(ADT_KERNEL_BUFFER, ctor)
{
  const Shape shape{2, 4, 6, 3};
  auto buffer = make_buffer<int, NCHWLayout>(shape);

  ASSERT_EQ(buffer.shape().count(), shape.count());
  ASSERT_EQ(buffer.shape().depth(), shape.depth());
  ASSERT_EQ(buffer.shape().height(), shape.height());
  ASSERT_EQ(buffer.shape().width(), shape.width());
}

TEST(ADT_KERNEL_BUFFER, access)
{
  const Shape shape{2, 4, 6, 3};
  auto buffer = make_buffer<int, NCHWLayout>(shape);

  ASSERT_EQ(buffer.at(1, 3, 5, 2), 0);
  buffer.at(1, 3, 5, 2) = 4;

  // Casting is introduced to use 'const T &at(...) const' method
  ASSERT_EQ(static_cast<const Buffer<int> &>(buffer).at(1, 3, 5, 2), 4);
}
