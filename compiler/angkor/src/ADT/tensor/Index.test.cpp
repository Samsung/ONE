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

  ASSERT_EQ(0, index.rank());
}

TEST(ADT_TENSOR_INDEX, ctor_initializer_list)
{
  const nncc::core::ADT::tensor::Index index{1, 3, 5, 7};

  ASSERT_EQ(4, index.rank());

  ASSERT_EQ(1, index.at(0));
  ASSERT_EQ(3, index.at(1));
  ASSERT_EQ(5, index.at(2));
  ASSERT_EQ(7, index.at(3));
}

TEST(ADT_TENSOR_INDEX, operator_add)
{
  nncc::core::ADT::tensor::Index index1{1, 2, 3, 4};
  nncc::core::ADT::tensor::Index index2{5, 6, 7, 8};
  nncc::core::ADT::tensor::Index result{index1 + index2};

  ASSERT_EQ(6, result.at(0));
  ASSERT_EQ(8, result.at(1));
  ASSERT_EQ(10, result.at(2));
  ASSERT_EQ(12, result.at(3));
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

  ASSERT_EQ(4, index.rank());
}

TEST(ADT_TENSOR_INDEX, at)
{
  nncc::core::ADT::tensor::Index index;

  index.resize(4);

  uint32_t indices[4] = {3, 5, 2, 7};

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    index.at(axis) = indices[axis];
    ASSERT_EQ(indices[axis], index.at(axis));
  }
}

TEST(ADT_TENSOR_INDEX, copy)
{
  const nncc::core::ADT::tensor::Index original{3, 5, 2, 7};
  const nncc::core::ADT::tensor::Index copied{original};

  ASSERT_EQ(copied.rank(), original.rank());

  for (uint32_t axis = 0; axis < 4; ++axis)
  {
    ASSERT_EQ(copied.at(axis), original.at(axis));
  }
}

TEST(ADT_TENSOR_INDEX, fill)
{
  nncc::core::ADT::tensor::Index index{1, 6};

  index.fill(3);

  ASSERT_EQ(2, index.rank());

  ASSERT_EQ(3, index.at(0));
  ASSERT_EQ(3, index.at(1));
}
