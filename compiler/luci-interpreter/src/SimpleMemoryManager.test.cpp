/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci_interpreter/SimpleMemoryManager.h"
#include <gtest/gtest.h>

using namespace luci_interpreter;
using namespace testing;

TEST(SimpleMemoryManager, basic)
{
  SimpleMemoryManager smm;
  Tensor t(DataType::U8, Shape({1, 16, 16, 256}), AffineQuantization{}, "t");

  EXPECT_NO_THROW(smm.allocate_memory(t));
  EXPECT_NO_THROW(smm.release_memory(t));
}

TEST(SimpleMemoryManager, huge)
{
  SimpleMemoryManager smm;
  Tensor t(DataType::U8, Shape({1, 512, 512, 256 * 3 * 3 * 4}), AffineQuantization{}, "t");

  EXPECT_NO_THROW(smm.allocate_memory(t));
  EXPECT_NO_THROW(smm.release_memory(t));
}

TEST(SimpleMemoryManager, string_dtype_NEG)
{
  SimpleMemoryManager smm;
  Tensor t(DataType::STRING, Shape({1, 16, 16, 4}), AffineQuantization{}, "t");

  EXPECT_ANY_THROW(smm.allocate_memory(t));
}

TEST(SimpleMemoryManager, negative_shape_NEG)
{
  SimpleMemoryManager smm;
  Tensor t(DataType::U8, Shape({1, 16, 16, -4}), AffineQuantization{}, "t");

  EXPECT_ANY_THROW(smm.allocate_memory(t));
}
