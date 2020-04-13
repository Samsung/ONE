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

#include "nest/Level.h"

#include <gtest/gtest.h>

TEST(LEVEL, constructor)
{
  nest::Level lv{3};

  ASSERT_EQ(lv.value(), 3);
}

TEST(LEVEL, operator_eq)
{
  ASSERT_TRUE(nest::Level(3) == nest::Level(3));
  ASSERT_FALSE(nest::Level(3) == nest::Level(4));
}

TEST(LEVEL, operator_lt)
{
  ASSERT_FALSE(nest::Level(3) < nest::Level(3));
  ASSERT_TRUE(nest::Level(3) < nest::Level(4));
  ASSERT_FALSE(nest::Level(4) < nest::Level(3));
}
