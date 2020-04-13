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

#include "nncc/core/ADT/tensor/Index.h"

#include <gtest/gtest.h>

TEST(ADT_TENSOR_INDEX, ctor)
{
  nncc::core::ADT::tensor::Index index;

  ASSERT_EQ(index.rank(), 0);
}

TEST(ADT_TENSOR_INDEX, ctor_initializer_list)
{
  const nncc::core::ADT::tensor::Index index{1, 3, 5, 7};

  ASSERT_EQ(index.rank(), 4);

  ASSERT_EQ(index.at(0), 1);
  ASSERT_EQ(index.at(1), 3);
  ASSERT_EQ(index.at(2), 5);
  ASSERT_EQ(index.at(3), 7);
}

TEST(ADT_TENSOR_INDEX, operator_add)
{
  nncc::core::ADT::tensor::Index index1{1, 2, 3, 4};
  nncc::core::ADT::tensor::Index index2{5, 6, 7, 8};
  nncc::core::ADT::tensor::Index result{index1 + index2};

  ASSERT_EQ(result.at(0), 6);
  ASSERT_EQ(result.at(1), 8);
  ASSERT_EQ(result.at(2), 10);
  ASSERT_EQ(result.at(3), 12);
}

TEST(ADT_TENSOR_INDEX, operator_eqaul)
{
  nncc::core::ADT::tensor::Index index1{1, 2, 3, 4};
  nncc::core::ADT::tensor::Index index2{1, 2, 3, 4};
  nncc::core::ADT::tensor::Index index3{5, 6, 7, 8};
  nncc::core::ADT::tensor::Index index4{1, 2};

  ASSERT_TRUE(index1 == index2);
  ASSERT_FALSE(index1 == index3);
  ASSERT_FALSE(index1 == index4);
}

TEST(ADT_TENSOR_INDEX, operator_add_different_size)
{
  nncc::core::ADT::tensor::Index index1{1, 2, 3, 4};
  nncc::core::ADT::tensor::Index index2{5, 6};

  EXPECT_THROW(index1 + index2, std::runtime_error);
}

TEST(ADT_TENSOR_INDEX, resize)
{
  nncc::core::ADT::tensor::Index index;

  index.resize(4);

  ASSERT_EQ(index.rank(), 4);
}

TEST(ADT_TENSOR_INDEX, at)
{
  nncc::core::ADT::tensor::Index index;

  index.resize(4);

  uint32_t indices[4] = {3, 5, 2, 7};

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    index.at(axis) = indices[axis];
    ASSERT_EQ(index.at(axis), indices[axis]);
  }
}

TEST(ADT_TENSOR_INDEX, copy)
{
  const nncc::core::ADT::tensor::Index original{3, 5, 2, 7};
  const nncc::core::ADT::tensor::Index copied{original};

  ASSERT_EQ(original.rank(), copied.rank());

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    ASSERT_EQ(original.at(axis), copied.at(axis));
  }
}

TEST(ADT_TENSOR_INDEX, fill)
{
  nncc::core::ADT::tensor::Index index{1, 6};

  index.fill(3);

  ASSERT_EQ(index.rank(), 2);

  ASSERT_EQ(index.at(0), 3);
  ASSERT_EQ(index.at(1), 3);
}
