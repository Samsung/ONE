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

#include "CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleDialectTest, get)
{
  using locoex::CircleDialect;

  auto d = CircleDialect::get();

  // get() SHOULD return a valid(non-null) pointer
  ASSERT_NE(d, nullptr);
  // The return value SHOULD be stable across multiple invocations
  ASSERT_EQ(CircleDialect::get(), d);
}
