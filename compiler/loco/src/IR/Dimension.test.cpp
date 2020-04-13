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

#include "loco/IR/Dimension.h"

#include <gtest/gtest.h>

namespace
{

struct DimensionTest : public ::testing::Test
{
protected:
  uint32_t value(void) const { return _value; }

private:
  uint32_t const _value{3};
};

} // namespace

TEST_F(DimensionTest, default_constructor)
{
  loco::Dimension dim;

  ASSERT_FALSE(dim.known());
}

TEST_F(DimensionTest, value_constructor)
{
  loco::Dimension dim{value()};

  ASSERT_TRUE(dim.known());
  ASSERT_EQ(dim.value(), value());
}

TEST_F(DimensionTest, set)
{
  loco::Dimension dim;

  dim.set(value());

  ASSERT_TRUE(dim.known());
  ASSERT_EQ(dim.value(), value());
}

TEST_F(DimensionTest, unset)
{
  loco::Dimension dim{value()};

  dim.unset();

  ASSERT_FALSE(dim.known());
}

TEST_F(DimensionTest, operator_eq)
{
  loco::Dimension unknown;
  loco::Dimension known{3};

  // Compare uint32_t and an unknown dimension
  ASSERT_FALSE(unknown == 3);
  ASSERT_FALSE(3 == unknown);

  // Compare uint32_t and a known dimension
  ASSERT_TRUE(known == 3);
  ASSERT_TRUE(3 == known);

  ASSERT_FALSE(known == 4);
  ASSERT_FALSE(4 == known);

  // Compare two known dimensions
  loco::Dimension another_known{3};
  ASSERT_TRUE(known == another_known);

  // Compare two unknown dimensions
  loco::Dimension unknown_a, unknown_b;
  ASSERT_TRUE(unknown_a.known() == false && unknown_b.known() == false);
  ASSERT_FALSE(unknown_a == unknown_b);
}

TEST_F(DimensionTest, make_unknown_dimension)
{
  auto dim = loco::make_dimension();

  ASSERT_FALSE(dim.known());
}
