/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "loco/IR/DepthwiseFilterIndex.h"

#include <gtest/gtest.h>

TEST(DepthwiseFilterIndexTest, default_constructor)
{
  loco::DepthwiseFilterIndex index;

  // All the values are 0 at the beginning
  ASSERT_EQ(0, index.channel());
  ASSERT_EQ(0, index.nth());
  ASSERT_EQ(0, index.row());
  ASSERT_EQ(0, index.column());
}

TEST(DepthwiseFilterIndexTest, settet_and_getter)
{
  loco::DepthwiseFilterIndex index;

  // Set depth
  index.channel() = 2;

  ASSERT_EQ(2, index.channel());
  ASSERT_EQ(0, index.nth());
  ASSERT_EQ(0, index.row());
  ASSERT_EQ(0, index.column());

  // Set multiplier
  index.nth() = 3;

  ASSERT_EQ(2, index.channel());
  ASSERT_EQ(3, index.nth());
  ASSERT_EQ(0, index.row());
  ASSERT_EQ(0, index.column());

  // Set height
  index.row() = 4;

  ASSERT_EQ(2, index.channel());
  ASSERT_EQ(3, index.nth());
  ASSERT_EQ(4, index.row());
  ASSERT_EQ(0, index.column());

  // Set width
  index.column() = 5;

  ASSERT_EQ(2, index.channel());
  ASSERT_EQ(3, index.nth());
  ASSERT_EQ(4, index.row());
  ASSERT_EQ(5, index.column());
}
