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

#include "loco/IR/Window.h"

#include <gtest/gtest.h>

TEST(WindowTest, default_constructor_2D)
{
  loco::Window<2> window;

  ASSERT_EQ(1, window.vertical());
  ASSERT_EQ(1, window.horizontal());
}

TEST(WindowTest, setter_and_getter_2D)
{
  loco::Window<2> window;

  window.vertical(2);

  ASSERT_EQ(2, window.vertical());
  ASSERT_EQ(1, window.horizontal());

  window.horizontal(3);

  ASSERT_EQ(2, window.vertical());
  ASSERT_EQ(3, window.horizontal());
}
