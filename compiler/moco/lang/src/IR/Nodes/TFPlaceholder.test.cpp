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

#include "moco/IR/Nodes/TFPlaceholder.h"
#include "moco/IR/TFDialect.h"

#include <gtest/gtest.h>

TEST(TFPlaceholderTest, constructor)
{
  moco::TFPlaceholder placeholder;

  ASSERT_EQ(placeholder.dialect(), moco::TFDialect::get());
  ASSERT_EQ(placeholder.opcode(), moco::TFOpcode::Placeholder);

  ASSERT_EQ(placeholder.dtype(), loco::DataType::Unknown);
  ASSERT_EQ(placeholder.rank(), 0);

  placeholder.dtype(loco::DataType::FLOAT32);
  ASSERT_EQ(placeholder.dtype(), loco::DataType::FLOAT32);

  placeholder.rank(2);
  ASSERT_EQ(placeholder.rank(), 2);

  placeholder.dim(0) = 2;
  placeholder.dim(1) = 3;

  ASSERT_TRUE(placeholder.dim(0).known());
  ASSERT_TRUE(placeholder.dim(1).known());

  ASSERT_EQ(placeholder.dim(0), 2);
  ASSERT_EQ(placeholder.dim(1), 3);
}
