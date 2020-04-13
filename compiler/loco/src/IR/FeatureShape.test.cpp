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

#include "loco/IR/FeatureShape.h"

#include <gtest/gtest.h>

TEST(FeatureShapeTest, default_constructor)
{
  loco::FeatureShape shape;

  ASSERT_FALSE(shape.count().known());
  ASSERT_FALSE(shape.depth().known());
  ASSERT_FALSE(shape.height().known());
  ASSERT_FALSE(shape.width().known());
}

TEST(FeatureShapeTest, settet_and_getter)
{
  loco::FeatureShape shape;

  // Set count
  shape.count() = 2;

  ASSERT_TRUE(shape.count().known());
  ASSERT_FALSE(shape.depth().known());
  ASSERT_FALSE(shape.height().known());
  ASSERT_FALSE(shape.width().known());

  ASSERT_EQ(shape.count(), 2);

  // Set depth
  shape.depth() = 3;

  ASSERT_TRUE(shape.count().known());
  ASSERT_TRUE(shape.depth().known());
  ASSERT_FALSE(shape.height().known());
  ASSERT_FALSE(shape.width().known());

  ASSERT_EQ(shape.count(), 2);
  ASSERT_EQ(shape.depth(), 3);

  // Set height
  shape.height() = 4;

  ASSERT_TRUE(shape.count().known());
  ASSERT_TRUE(shape.depth().known());
  ASSERT_TRUE(shape.height().known());
  ASSERT_FALSE(shape.width().known());

  ASSERT_EQ(shape.count(), 2);
  ASSERT_EQ(shape.depth(), 3);
  ASSERT_EQ(shape.height(), 4);

  // Set width
  shape.width() = 5;

  ASSERT_TRUE(shape.count().known());
  ASSERT_TRUE(shape.depth().known());
  ASSERT_TRUE(shape.height().known());
  ASSERT_TRUE(shape.width().known());

  ASSERT_EQ(shape.count(), 2);
  ASSERT_EQ(shape.depth(), 3);
  ASSERT_EQ(shape.height(), 4);
  ASSERT_EQ(shape.width(), 5);
}
