/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "loco/IR/TensorIndex.h"

#include <gtest/gtest.h>
#include <stdexcept>

TEST(TensorIndexTest, default_constructor)
{
  loco::TensorIndex index;
  ASSERT_EQ(index.rank(), 0);
}

TEST(TensorIndexTest, resize_increase_rank)
{
  loco::TensorIndex index;
  index.resize(3);
  ASSERT_EQ(index.rank(), 3);
}

TEST(TensorIndexTest, resize_to_zero)
{
  loco::TensorIndex index;
  index.resize(3);
  ASSERT_EQ(index.rank(), 3);
  index.resize(0);
  ASSERT_EQ(index.rank(), 0);
}

TEST(TensorIndexTest, resize_decrease_rank)
{
  loco::TensorIndex index;
  index.resize(5);
  ASSERT_EQ(index.rank(), 5);
  index.resize(2);
  ASSERT_EQ(index.rank(), 2);
}

TEST(TensorIndexTest, at_set_and_get)
{
  loco::TensorIndex index;
  index.resize(3);

  index.at(0) = 10;
  index.at(1) = 20;
  index.at(2) = 30;

  ASSERT_EQ(index.at(0), 10);
  ASSERT_EQ(index.at(1), 20);
  ASSERT_EQ(index.at(2), 30);
}

TEST(TensorIndexTest, at_const_version)
{
  loco::TensorIndex index;
  index.resize(2);
  index.at(0) = 5;
  index.at(1) = 15;

  const loco::TensorIndex &const_index = index;

  ASSERT_EQ(const_index.at(0), 5);
  ASSERT_EQ(const_index.at(1), 15);
}

TEST(TensorIndexTest, at_out_of_range_after_resize_NEG)
{
  loco::TensorIndex index;
  index.resize(2);
  ASSERT_EQ(index.rank(), 2);

  EXPECT_THROW(index.at(2), std::out_of_range);
  EXPECT_THROW(index.at(3), std::out_of_range);
}

TEST(TensorIndexTest, at_out_of_range_on_empty_index_NEG)
{
  loco::TensorIndex index;
  ASSERT_EQ(index.rank(), 0);

  EXPECT_THROW(index.at(0), std::out_of_range);
}

TEST(TensorIndexTest, at_const_out_of_range_NEG)
{
  loco::TensorIndex index;
  index.resize(1);
  const loco::TensorIndex &const_index = index;

  EXPECT_THROW(const_index.at(1), std::out_of_range);
}
