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

#include "nncc/core/ADT/tensor/Buffer.h"
#include "nncc/core/ADT/tensor/LexicalLayout.h"

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Buffer;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::Shape;

using nncc::core::ADT::tensor::make_buffer;

TEST(ADT_TENSOR_BUFFER, ctor)
{
  const Shape shape{2, 3};
  auto buffer = make_buffer<int, LexicalLayout>(shape);

  ASSERT_EQ(shape, buffer.shape());
}

TEST(ADT_TENSOR_BUFFER, access)
{
  const Shape shape{2, 3};
  auto buffer = make_buffer<int, LexicalLayout>(shape);

  const Index index{1, 2};

  ASSERT_EQ(0, buffer.at(index));
  buffer.at(index) = 4;

  // Casting is introduced to use 'const T &at(...) const' method
  ASSERT_EQ(4, static_cast<const Buffer<int> &>(buffer).at(index));
}
