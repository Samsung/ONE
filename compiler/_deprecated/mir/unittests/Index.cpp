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

#include <gtest/gtest.h>

#include "mir/Shape.h"
#include "mir/Index.h"

using namespace mir;

TEST(Shape, Base)
{
  Shape s1{3, 2};
  ASSERT_EQ(s1.rank(), 2);
  ASSERT_EQ(s1.dim(0), 3);
  ASSERT_EQ(s1.dim(1), 2);
  ASSERT_EQ(s1.dim(-1), 2);
  ASSERT_EQ(s1.dim(-2), 3);
  ASSERT_EQ(s1.numElements(), 6);

  s1.dim(1) = 4;
  ASSERT_EQ(s1.dim(1), 4);
  ASSERT_EQ(s1.numElements(), 12);

  Shape s2 = s1;
  ASSERT_EQ(s1, s2);

  s2.resize(1);
  ASSERT_NE(s1, s2);

  s2.resize(2);
  s2.dim(1) = s1.dim(1);
  ASSERT_EQ(s1, s2);
}

TEST(Index, Base)
{
  Index idx{3, 2};
  ASSERT_EQ(idx.rank(), 2);
  ASSERT_EQ(idx.at(0), 3);
  ASSERT_EQ(idx.at(1), 2);
  ASSERT_EQ(idx.at(-1), 2);
  ASSERT_EQ(idx.at(-2), 3);

  idx.at(1) = 4;
  ASSERT_EQ(idx.at(1), 4);

  idx.resize(1);
  ASSERT_EQ(idx.rank(), 1);
}
