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

#include "moco/IR/Nodes/TFStridedSlice.h"
#include "moco/IR/TFDialect.h"

#include <gtest/gtest.h>

TEST(TFStridedSliceTest, constructor)
{
  moco::TFStridedSlice node;

  ASSERT_EQ(node.dialect(), moco::TFDialect::get());
  ASSERT_EQ(node.opcode(), moco::TFOpcode::StridedSlice);

  ASSERT_EQ(node.input(), nullptr);
  ASSERT_EQ(node.begin(), nullptr);
  ASSERT_EQ(node.end(), nullptr);
  ASSERT_EQ(node.strides(), nullptr);
  ASSERT_EQ(node.begin_mask(), 0);
  ASSERT_EQ(node.end_mask(), 0);
  ASSERT_EQ(node.ellipsis_mask(), 0);
  ASSERT_EQ(node.new_axis_mask(), 0);
  ASSERT_EQ(node.shrink_axis_mask(), 0);
}
