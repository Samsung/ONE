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

#include "ArrayIndex.h"

#include <gtest/gtest.h>

TEST(LuciPassHelpersArrayIndex, array_index_4d)
{
  luci::Array4DIndex idx(5, 4, 3, 2);

  EXPECT_EQ(idx(0, 0, 0, 0), 0);

  // stride
  EXPECT_EQ(idx(1, 0, 0, 0), idx.stride(0));
  EXPECT_EQ(idx(0, 1, 0, 0), idx.stride(1));
  EXPECT_EQ(idx(0, 0, 1, 0), idx.stride(2));
  EXPECT_EQ(idx(0, 0, 0, 1), idx.stride(3));

  // size
  EXPECT_EQ(idx.size(), 5 * 4 * 3 * 2);

  EXPECT_EQ(idx(4, 3, 2, 1), 4 * 4 * 3 * 2 + 3 * 3 * 2 + 2 * 2 + 1);
}

TEST(LuciPassHelpersArrayIndex, array_invalid_index_4d_NEG)
{
  luci::Array4DIndex idx(4, 4, 3, 2);

  EXPECT_ANY_THROW(idx(5, 0, 0, 0));
}

TEST(LuciPassHelpersArrayIndex, array_invalid_dim_4d_NEG)
{
  EXPECT_ANY_THROW(luci::Array4DIndex idx(4, 0, 3, 2));
}
