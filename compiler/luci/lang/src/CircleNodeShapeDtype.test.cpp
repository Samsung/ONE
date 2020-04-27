/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/CircleNodes.h"

#include <gtest/gtest.h>

TEST(CircleNodeShapeDTypeTest, constructor)
{
  luci::CircleAdd node;

  ASSERT_EQ(loco::DataType::Unknown, node.dtype());
  ASSERT_EQ(0, node.rank());
}

TEST(CircleNodeShapeDTypeTest, values)
{
  luci::CircleAdd node;

  node.dtype(loco::DataType::FLOAT32);
  ASSERT_EQ(loco::DataType::FLOAT32, node.dtype());

  node.rank(4);
  ASSERT_EQ(4, node.rank());
  ASSERT_FALSE(node.dim(0).known());

  node.dim(0) = loco::Dimension(1);
  ASSERT_TRUE(node.dim(0).known());
}

TEST(CircleNodeShapeDTypeTest, values_NEG)
{
  luci::CircleAdd node;

  node.rank(4);
  EXPECT_ANY_THROW(node.dim(100).known());
  EXPECT_ANY_THROW(node.dim(100) = loco::Dimension(1));
}
