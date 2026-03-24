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

#include "coco/IR/Stride2D.h"

#include <gtest/gtest.h>

TEST(IR_STRIDE_2D, default_constructor)
{
  coco::Stride2D stride;

  ASSERT_EQ(stride.vertical(), 1);
  ASSERT_EQ(stride.horizontal(), 1);
}

TEST(IR_STRIDE_2D, explicit_constructor_4)
{
  coco::Stride2D stride{2, 3};

  ASSERT_EQ(stride.vertical(), 2);
  ASSERT_EQ(stride.horizontal(), 3);
}

TEST(IR_STRIDE_2D, update)
{
  coco::Stride2D stride;

  stride.vertical(2).horizontal(3);

  ASSERT_EQ(stride.vertical(), 2);
  ASSERT_EQ(stride.horizontal(), 3);
}
