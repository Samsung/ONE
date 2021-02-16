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

#include "FuseActivationFunctionPassInternal.h"

#include "luci/Pass/FuseActivationFunctionPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 *         [Conv1]
 *           |
 *     [Activation func]
 *           |
 *         [Conv2]
 *
 *  AFTER
 *
 *   [Conv1 + Activation func]
 *           |
 *         [Conv2]
 *
 */
class SimpleGraph
{
public:
  SimpleGraph()
  {
    conv1 = g.nodes()->create<luci::CircleConv2D>();
    conv2 = g.nodes()->create<luci::CircleConv2D>();
    relu = g.nodes()->create<luci::CircleRelu>();

    conv1->fusedActivationFunction(luci::FusedActFunc::NONE);

    relu->features(conv1);
    conv2->input(relu);

    conv1->name("conv1");
    conv2->name("conv2");
    relu->name("relu");
  }

public:
  loco::Graph g;
  luci::CircleConv2D *conv1;
  luci::CircleConv2D *conv2;
  luci::CircleRelu *relu;
};

} // namespace

TEST(FuseActivationFunctionPassTest, name)
{
  luci::FuseActivationFunctionPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FusePreActivationBatchNorm, fuse_activation_function)
{
  SimpleGraph g;

  EXPECT_TRUE(luci::fuse_activation_function(g.relu));

  EXPECT_EQ(g.conv1, g.conv2->input());
}

TEST(FusePreActivationBatchNorm, fuse_activation_function_dup_relu)
{
  SimpleGraph g;
  g.conv1->fusedActivationFunction(luci::FusedActFunc::RELU);

  EXPECT_TRUE(luci::fuse_activation_function(g.relu));

  EXPECT_EQ(g.conv1, g.conv2->input());
}

TEST(FusePreActivationBatchNorm, fuse_activation_function_NEG)
{
  SimpleGraph g;
  g.conv2->input(g.conv1);

  // Conv1 has multiple successors
  EXPECT_FALSE(luci::fuse_activation_function(g.relu));

  g.conv2->input(g.relu);
  g.conv1->fusedActivationFunction(luci::FusedActFunc::TANH);

  // Conv1 already has activation function
  EXPECT_FALSE(luci::fuse_activation_function(g.relu));
}
