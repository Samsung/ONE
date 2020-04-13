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

#include "coco/IR/Window2D.h"

#include <gtest/gtest.h>

TEST(IR_WINDOW_2D, default_constructor)
{
  coco::Window2D window;

  ASSERT_EQ(window.height(), 1);
  ASSERT_EQ(window.width(), 1);
}

TEST(IR_WINDOW_2D, explicit_constructor_4)
{
  coco::Window2D window{2, 3};

  ASSERT_EQ(window.height(), 2);
  ASSERT_EQ(window.width(), 3);
}

TEST(IR_WINDOW_2D, update)
{
  coco::Window2D window;

  window.height(2);
  window.width(3);

  ASSERT_EQ(window.height(), 2);
  ASSERT_EQ(window.width(), 3);
}
