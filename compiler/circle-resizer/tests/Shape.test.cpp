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

#include "Shape.h"

#include <gtest/gtest.h>

using namespace circle_resizer;

TEST(ShapeTest, create_scalar)
{
  auto const scalar = Shape::scalar();
  EXPECT_TRUE(scalar.is_scalar());
  EXPECT_EQ(scalar.rank(), 0);
}

TEST(ShapeTest, gather_scalar_NEG)
{
  auto const scalar = Shape::scalar();
  EXPECT_THROW(scalar[0], std::invalid_argument);
}

TEST(ShapeTest, is_dynamic)
{
  EXPECT_FALSE(Shape::scalar().is_dynamic());

  const auto shape1 = Shape{1, 2};
  EXPECT_FALSE(shape1.is_dynamic());

  const auto shape2 = Shape{Dim{1}, Dim::dynamic()};
  EXPECT_TRUE(shape2.is_dynamic());

  const auto shape3 = Shape{Dim::dynamic(), Dim::dynamic()};
  EXPECT_TRUE(shape3.is_dynamic());

  const auto shape4 = Shape{2};
  EXPECT_FALSE(shape4.is_dynamic());
}

TEST(ShapeTest, index_operator_NEG)
{
  const auto shape = Shape{Dim{1}, Dim::dynamic()};
  EXPECT_THROW(shape[2], std::out_of_range);
}

TEST(ShapeTest, equal_operator)
{
  // static vs static with other rank
  auto shape1 = Shape{1, 2, 3};
  auto shape2 = Shape{1, 2};
  EXPECT_FALSE(shape1 == shape2);

  // different static vs static
  shape1 = Shape{1, 2, 3};
  shape2 = Shape{1, 3, 3};
  EXPECT_FALSE(shape1 == shape2);

  // the same dynamic vs dynamic
  shape1 = Shape{Dim::dynamic(), Dim::dynamic()};
  shape2 = Shape{Dim::dynamic(), Dim::dynamic()};
  EXPECT_TRUE(shape1 == shape2);

  shape1 = Shape{Dim::dynamic(), Dim{2}};
  shape2 = Shape{Dim::dynamic(), Dim{2}};
  EXPECT_TRUE(shape1 == shape2);

  // different dynamic vs dynamic
  shape1 = Shape{Dim::dynamic(), Dim::dynamic()};
  shape2 = Shape{Dim::dynamic(), Dim{2}};
  EXPECT_FALSE(shape1 == shape2);

  // static vs dynamic
  shape1 = Shape{Dim::dynamic(), Dim::dynamic()};
  shape2 = Shape{1, 2};
  EXPECT_FALSE(shape1 == shape2);

  // scalar vs scalar
  shape1 = Shape::scalar();
  shape2 = Shape::scalar();
  EXPECT_TRUE(shape1 == shape2);

  // scalar vs static
  shape1 = Shape{1};
  shape2 = Shape::scalar();
  EXPECT_FALSE(shape1 == shape2);
}

TEST(ShapeTest, print)
{
  { // scalar
    auto shape = Shape::scalar();
    std::stringstream ss;
    ss << shape;
    EXPECT_EQ(ss.str(), "[]");
  }

  { // 1D
    auto shape = Shape{1};
    std::stringstream ss;
    ss << shape;
    EXPECT_EQ(ss.str(), "[1]");
  }

  { // static
    auto shape = Shape{1, 2, 3};
    std::stringstream ss;
    ss << shape;
    EXPECT_EQ(ss.str(), "[1, 2, 3]");
  }

  { // dynamic
    auto shape = Shape{Dim{1}, Dim::dynamic(), Dim{3}};
    std::stringstream ss;
    ss << shape;
    EXPECT_EQ(ss.str(), "[1, -1, 3]");
  }

  { // all dimensions dynamic
    auto shape = Shape{Dim::dynamic(), Dim::dynamic()};
    std::stringstream ss;
    ss << shape;
    EXPECT_EQ(ss.str(), "[-1, -1]");
  }
}
