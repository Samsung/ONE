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

#include "stdex/Set.h"

#include <gtest/gtest.h>

TEST(SET, operator_eq)
{
  ASSERT_TRUE(std::set<int>({1, 2, 3}) == std::set<int>({1, 2, 3}));
  ASSERT_FALSE(std::set<int>({1, 3}) == std::set<int>({1, 2, 3}));
}

TEST(SET, operator_diff)
{
  const std::set<int> lhs{1, 2, 3};
  const std::set<int> rhs{2, 4};

  auto res = lhs - rhs;

  ASSERT_EQ(res.size(), 2);
  ASSERT_NE(res.find(1), res.end());
  ASSERT_NE(res.find(3), res.end());
}
