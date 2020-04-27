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

#include "nncc/core/ADT/tensor/LexicalLayout.h"

#include <type_traits>

#include <gtest/gtest.h>

TEST(ADT_TENSOR_LEXICAL_LAYOUT, last)
{
  const nncc::core::ADT::tensor::Shape shape{4, 3, 6};
  const nncc::core::ADT::tensor::Index curr{1, 1, 1};
  const nncc::core::ADT::tensor::Index next{1, 1, 2};

  const nncc::core::ADT::tensor::LexicalLayout l;

  ASSERT_EQ(l.offset(shape, next), l.offset(shape, curr) + 1);
}

TEST(ADT_TENSOR_LEXICAL_LAYOUT, lexical_middle)
{
  const nncc::core::ADT::tensor::Shape shape{4, 3, 6};
  const nncc::core::ADT::tensor::Index curr{1, 1, 1};
  const nncc::core::ADT::tensor::Index next{1, 2, 1};

  const nncc::core::ADT::tensor::LexicalLayout l;

  ASSERT_EQ(l.offset(shape, next), l.offset(shape, curr) + 6);
}

TEST(ADT_TENSOR_LEXICAL_LAYOUT, lexical_first)
{
  const nncc::core::ADT::tensor::Shape shape{4, 3, 6};
  const nncc::core::ADT::tensor::Index curr{1, 1, 1};
  const nncc::core::ADT::tensor::Index next{2, 1, 1};

  const nncc::core::ADT::tensor::LexicalLayout l;

  ASSERT_EQ(l.offset(shape, next), l.offset(shape, curr) + 6 * 3);
}
