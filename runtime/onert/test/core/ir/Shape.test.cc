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

#include <ir/Shape.h>

#include <gtest/gtest.h>

TEST(ShapeTest, basic_test)
{
  {
    onert::ir::Shape shape(3);

    shape.dim(0) = 1;
    shape.dim(1) = 2;
    shape.dim(2) = 3;

    ASSERT_EQ(shape.rank(), 3);
    ASSERT_EQ(shape.num_elements(), 6);
    ASSERT_EQ(onert::ir::rankMaybeUnspecified(shape), false);
    ASSERT_EQ(shape.hasUnspecifiedDims(), false);
  }
  {
    onert::ir::Shape shape; // scalar or rank is unspecified

    ASSERT_EQ(shape.rank(), 0);
    ASSERT_EQ(shape.num_elements(), 1);
    ASSERT_EQ(onert::ir::rankMaybeUnspecified(shape), true);
    ASSERT_EQ(shape.hasUnspecifiedDims(), false);
  }
}

TEST(ShapeTest, neg_basic_test)
{
  {
    onert::ir::Shape shape(2);

    shape.dim(0) = 1;
    shape.dim(1) = onert::ir::Shape::UNSPECIFIED_DIM;

    ASSERT_EQ(shape.rank(), 2);
    ASSERT_EQ(onert::ir::rankMaybeUnspecified(shape), false);
    ASSERT_EQ(shape.hasUnspecifiedDims(), true);
    EXPECT_ANY_THROW(shape.num_elements());
  }
}
