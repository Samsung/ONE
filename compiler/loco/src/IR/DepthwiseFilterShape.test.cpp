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

#include "loco/IR/DepthwiseFilterShape.h"

#include <gtest/gtest.h>

TEST(DepthwiseFilterShapeTest, default_constructor)
{
  loco::DepthwiseFilterShape shape;

  ASSERT_FALSE(shape.depth().known());
  ASSERT_FALSE(shape.multiplier().known());
  ASSERT_FALSE(shape.height().known());
  ASSERT_FALSE(shape.width().known());
}

TEST(DepthwiseFilterShapeTest, settet_and_getter)
{
  loco::DepthwiseFilterShape shape;

  // Set depth
  shape.depth() = 2;

  ASSERT_TRUE(shape.depth().known());
  ASSERT_FALSE(shape.multiplier().known());
  ASSERT_FALSE(shape.height().known());
  ASSERT_FALSE(shape.width().known());

  ASSERT_EQ(2, shape.depth());

  // Set multiplier
  shape.multiplier() = 3;

  ASSERT_TRUE(shape.depth().known());
  ASSERT_TRUE(shape.multiplier().known());
  ASSERT_FALSE(shape.height().known());
  ASSERT_FALSE(shape.width().known());

  ASSERT_EQ(2, shape.depth());
  ASSERT_EQ(3, shape.multiplier());

  // Set height
  shape.height() = 4;

  ASSERT_TRUE(shape.depth().known());
  ASSERT_TRUE(shape.multiplier().known());
  ASSERT_TRUE(shape.height().known());
  ASSERT_FALSE(shape.width().known());

  ASSERT_EQ(2, shape.depth());
  ASSERT_EQ(3, shape.multiplier());
  ASSERT_EQ(4, shape.height());

  // Set width
  shape.width() = 5;

  ASSERT_TRUE(shape.depth().known());
  ASSERT_TRUE(shape.multiplier().known());
  ASSERT_TRUE(shape.height().known());
  ASSERT_TRUE(shape.width().known());

  ASSERT_EQ(2, shape.depth());
  ASSERT_EQ(3, shape.multiplier());
  ASSERT_EQ(4, shape.height());
  ASSERT_EQ(5, shape.width());
}
