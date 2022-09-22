/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "StringUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/AttrFusedActFunc.h>

#include <gtest/gtest.h>

using namespace dalgona;

TEST(DalgonaUtilTest, toString_basic)
{
  luci::CircleConv2D node;

  EXPECT_EQ("Conv2D", toString(node.opcode()));
}

TEST(DalgonaUtilTest, toString_fused_act_func)
{
  EXPECT_EQ("undefined", toString(luci::FusedActFunc::UNDEFINED));
  EXPECT_EQ("none", toString(luci::FusedActFunc::NONE));
  EXPECT_EQ("relu", toString(luci::FusedActFunc::RELU));
  EXPECT_EQ("relu6", toString(luci::FusedActFunc::RELU6));
  EXPECT_EQ("relu_n1_to_1", toString(luci::FusedActFunc::RELU_N1_TO_1));
  EXPECT_EQ("tanh", toString(luci::FusedActFunc::TANH));
  EXPECT_EQ("sign_bit", toString(luci::FusedActFunc::SIGN_BIT));
}
