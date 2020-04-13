/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/Nodes/CircleDepthwiseConv2D.h"

#include "luci/IR/CircleDialect.h"

#include <gtest/gtest.h>

TEST(CircleDepthwiseConv2DTest, constructor_P)
{
  luci::CircleDepthwiseConv2D dw_conv2d_node;

  ASSERT_EQ(dw_conv2d_node.dialect(), luci::CircleDialect::get());
  ASSERT_EQ(dw_conv2d_node.opcode(), luci::CircleOpcode::DEPTHWISE_CONV_2D);

  ASSERT_EQ(dw_conv2d_node.input(), nullptr);
  ASSERT_EQ(dw_conv2d_node.filter(), nullptr);
  ASSERT_EQ(dw_conv2d_node.bias(), nullptr);
  ASSERT_EQ(dw_conv2d_node.padding(), luci::Padding::UNDEFINED);
  ASSERT_EQ(dw_conv2d_node.stride()->h(), 1);
  ASSERT_EQ(dw_conv2d_node.stride()->w(), 1);
  ASSERT_EQ(dw_conv2d_node.depthMultiplier(), 0);
  ASSERT_EQ(dw_conv2d_node.fusedActivationFunction(), luci::FusedActFunc::UNDEFINED);
}
