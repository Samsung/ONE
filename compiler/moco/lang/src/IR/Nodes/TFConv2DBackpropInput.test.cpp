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

#include "moco/IR/Nodes/TFConv2DBackpropInput.h"
#include "moco/IR/TFDialect.h"

#include <gtest/gtest.h>

TEST(TFConv2DBackpropInputTest, constructor)
{
  moco::TFConv2DBackpropInput conv2dbi_node;

  ASSERT_EQ(conv2dbi_node.dialect(), moco::TFDialect::get());
  ASSERT_EQ(conv2dbi_node.opcode(), moco::TFOpcode::Conv2DBackpropInput);

  ASSERT_EQ(conv2dbi_node.input_sizes(), nullptr);
  ASSERT_EQ(conv2dbi_node.filter(), nullptr);
  ASSERT_EQ(conv2dbi_node.out_backprop(), nullptr);
  ASSERT_EQ(conv2dbi_node.padding(), "");
  ASSERT_EQ(conv2dbi_node.data_layout(), "");
  ASSERT_EQ(conv2dbi_node.strides().size(), 0);
}
