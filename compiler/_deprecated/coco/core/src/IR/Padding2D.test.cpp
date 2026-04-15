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

#include "coco/IR/Padding2D.h"

#include <gtest/gtest.h>

TEST(IR_PADDING, default_constructor)
{
  coco::Padding2D pad;

  ASSERT_EQ(pad.top(), 0);
  ASSERT_EQ(pad.bottom(), 0);
  ASSERT_EQ(pad.left(), 0);
  ASSERT_EQ(pad.right(), 0);
}

TEST(IR_PADDING, explicit_constructor_4)
{
  coco::Padding2D pad{1, 2, 3, 4};

  ASSERT_EQ(pad.top(), 1);
  ASSERT_EQ(pad.bottom(), 2);
  ASSERT_EQ(pad.left(), 3);
  ASSERT_EQ(pad.right(), 4);
}

TEST(IR_PADDING, update)
{
  coco::Padding2D pad;

  pad.top(1).bottom(2).left(3).right(4);

  ASSERT_EQ(pad.top(), 1);
  ASSERT_EQ(pad.bottom(), 2);
  ASSERT_EQ(pad.left(), 3);
  ASSERT_EQ(pad.right(), 4);
}
