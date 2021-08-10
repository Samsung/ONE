/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci_interpreter/BuddyMemoryManager.h"
#include <gtest/gtest.h>

namespace luci_interpreter
{
namespace
{

using namespace testing;

TEST(BuddyMemoryManager, BuddyMemoryManager)
{
  uint8_t *memPool = new uint8_t[3 * 5000];
  auto buddy_memory_manager = std::make_unique<BuddyMemoryManager>(memPool, 5000);
  Tensor first_tensor(DataType::U8, Shape({5}), AffineQuantization{}, "first_tensor");

  buddy_memory_manager->allocate_memory(&first_tensor);

  uint8_t data_1[] = {1, 2, 3, 4, 5};

  first_tensor.writeData(data_1, 5);
  uint8_t array_1[5];
  first_tensor.readData(array_1, 5); // data<void>();
  for (int i = 0; i < 5; i++)
  {
    EXPECT_EQ(data_1[i], array_1[i]);
  }

  Tensor second_tensor(DataType::U8, Shape({2, 5}), AffineQuantization{}, "second_tensor");
  buddy_memory_manager->allocate_memory(&second_tensor);

  uint8_t data_2[2][5] = {{11, 22, 33, 44, 55}, {12, 23, 34, 45, 56}};
  second_tensor.writeData(data_2, 10);

  uint8_t array_2[2][5];
  second_tensor.readData(array_2, 10);
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 5; j++)
    {
      EXPECT_EQ(data_2[i][j], array_2[i][j]);
    }
  }

  buddy_memory_manager->release_memory(&first_tensor);
  EXPECT_EQ(first_tensor.data<void>(), nullptr);

  buddy_memory_manager->release_memory(&second_tensor);
  EXPECT_EQ(second_tensor.data<void>(), nullptr);
}

} // namespace
} // namespace luci_interpreter
