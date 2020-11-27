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

#include "nncc/core/ADT/tensor/Overlay.h"
#include "nncc/core/ADT/tensor/LexicalLayout.h"

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::Overlay;

using nncc::core::ADT::tensor::make_overlay;

TEST(ADT_TENSOR_OVERLAY, ctor)
{
  const Shape shape{2, 3};

  int data[2 * 3] = {
    0,
  };
  auto view = make_overlay<int, LexicalLayout>(shape, data);

  ASSERT_EQ(shape, view.shape());
}

TEST(ADT_TENSOR_OVERLAY, read)
{
  const Shape shape{2, 3};

  int data[2 * 3] = {
    0,
  };
  const auto view = make_overlay<int, LexicalLayout>(shape, data);

  LexicalLayout layout{};

  const Index index{1, 2};

  ASSERT_EQ(0, data[layout.offset(shape, index)]);
  data[layout.offset(shape, index)] = 2;
  ASSERT_EQ(2, view.at(index));
}

TEST(ADT_TENSOR_OVERLAY, access)
{
  const Shape shape{2, 3};

  int data[2 * 3] = {
    0,
  };
  auto view = make_overlay<int, LexicalLayout>(shape, data);

  LexicalLayout layout{};

  const Index index{1, 2};

  ASSERT_EQ(0, data[layout.offset(shape, index)]);
  view.at(index) = 4;
  ASSERT_EQ(4, data[layout.offset(shape, index)]);
}
