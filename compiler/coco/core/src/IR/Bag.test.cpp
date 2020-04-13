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

#include "coco/IR/Bag.h"

#include <gtest/gtest.h>

TEST(IR_BAG, ctor_should_set_size)
{
  coco::Bag b{3};

  ASSERT_EQ(b.size(), 3);

  // Bag has no read/updates at the beginning
  EXPECT_EQ(b.reads()->size(), 0);
  EXPECT_EQ(b.updates()->size(), 0);
}
