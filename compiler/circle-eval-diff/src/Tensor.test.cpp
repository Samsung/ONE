/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Tensor.h"

#include <gtest/gtest.h>

using Tensor = circle_eval_diff::Tensor;

namespace
{

template <loco::DataType DT> void test_out_of_buffer_range()
{
  Tensor t;

  t.shape({1, 2, 3});
  t.dtype(DT);
  t.size<DT>(6);

  EXPECT_ANY_THROW(t.at<DT>(6));
}

template <loco::DataType DT> void test_getter_setter()
{
  Tensor t;

  // Check shape
  t.shape({1, 2, 3});
  EXPECT_EQ(3, t.rank());
  EXPECT_EQ(1, t.dim(0));
  EXPECT_EQ(2, t.dim(1));
  EXPECT_EQ(3, t.dim(2));

  // Check dtype
  t.dtype(DT);
  EXPECT_EQ(DT, t.dtype());

  // Check buffer
  t.size<DT>(6);
  EXPECT_EQ(6 * sizeof(typename loco::DataTypeImpl<DT>::Type), t.byte_size());
  for (uint32_t i = 0; i < 6; i++)
    t.at<DT>(i) = i;

  for (uint32_t i = 0; i < 6; i++)
    EXPECT_EQ(i, t.at<DT>(i));
}

} // namespace

TEST(CircleEvalDiffTensorTest, constructor)
{
  Tensor t;

  EXPECT_EQ(0, t.byte_size());
  EXPECT_EQ(0, t.rank());
  EXPECT_EQ(loco::DataType::Unknown, t.dtype());
}

TEST(CircleEvalDiffTensorTest, getter_setter)
{
  test_getter_setter<loco::DataType::S64>();
  test_getter_setter<loco::DataType::S32>();
  test_getter_setter<loco::DataType::S16>();
  test_getter_setter<loco::DataType::U8>();
  test_getter_setter<loco::DataType::FLOAT32>();

  SUCCEED();
}

TEST(CircleEvalDiffTensorTest, out_of_shape_range_NEG)
{
  Tensor t;
  t.shape({1, 2, 2, 3});

  EXPECT_ANY_THROW(t.dim(4));
}

TEST(CircleEvalDiffTensorTest, out_of_buffer_range_NEG)
{
  test_out_of_buffer_range<loco::DataType::S64>();
  test_out_of_buffer_range<loco::DataType::S32>();
  test_out_of_buffer_range<loco::DataType::S16>();
  test_out_of_buffer_range<loco::DataType::U8>();
  test_out_of_buffer_range<loco::DataType::FLOAT32>();

  SUCCEED();
}
