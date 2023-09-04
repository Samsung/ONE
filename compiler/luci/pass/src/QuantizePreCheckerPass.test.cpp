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

#include "luci/Pass/QuantizePreCheckerPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

class SimpleConv2DGraph
{
public:
  SimpleConv2DGraph(bool make_valid)
  {
    conv2d_node = g.nodes()->create<luci::CircleConv2D>();
    input_1 = g.nodes()->create<luci::CircleInput>();
    filter = g.nodes()->create<luci::CircleConst>();

    conv2d_node->input(input_1);
    conv2d_node->filter(filter);

    if (make_valid)
    {
      bias = g.nodes()->create<luci::CircleConst>();
      conv2d_node->bias(bias);
    }
    else
    {
      input_2 = g.nodes()->create<luci::CircleInput>();
      conv2d_node->bias(input_2);
    }

    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    output->from(conv2d_node);
  }

public:
  loco::Graph g;

private:
  luci::CircleConv2D *conv2d_node = nullptr;
  luci::CircleInput *input_1 = nullptr;
  luci::CircleInput *input_2 = nullptr;
  luci::CircleConst *filter = nullptr;
  luci::CircleConst *bias = nullptr;
  luci::CircleOutput *output = nullptr;
};

class SimpleDepthConv2DGraph
{
public:
  SimpleDepthConv2DGraph(bool make_valid)
  {
    depth_conv2d_node = g.nodes()->create<luci::CircleDepthwiseConv2D>();
    input_1 = g.nodes()->create<luci::CircleInput>();
    filter = g.nodes()->create<luci::CircleConst>();

    depth_conv2d_node->input(input_1);
    depth_conv2d_node->filter(filter);

    if (make_valid)
    {
      bias = g.nodes()->create<luci::CircleConst>();
      depth_conv2d_node->bias(bias);
    }
    else
    {
      input_2 = g.nodes()->create<luci::CircleInput>();
      depth_conv2d_node->bias(input_2);
    }

    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    output->from(depth_conv2d_node);
  }

public:
  loco::Graph g;

private:
  luci::CircleDepthwiseConv2D *depth_conv2d_node = nullptr;
  luci::CircleInput *input_1 = nullptr;
  luci::CircleInput *input_2 = nullptr;
  luci::CircleConst *filter = nullptr;
  luci::CircleConst *bias = nullptr;
  luci::CircleOutput *output = nullptr;
};

class SimpleFCGraph
{
public:
  SimpleFCGraph(bool make_valid)
  {
    fc_node = g.nodes()->create<luci::CircleFullyConnected>();
    input_1 = g.nodes()->create<luci::CircleInput>();
    weights = g.nodes()->create<luci::CircleConst>();

    fc_node->input(input_1);
    fc_node->weights(weights);

    if (make_valid)
    {
      bias = g.nodes()->create<luci::CircleConst>();
      fc_node->bias(bias);
    }
    else
    {
      input_2 = g.nodes()->create<luci::CircleInput>();
      fc_node->bias(input_2);
    }

    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    output->from(fc_node);
  }

public:
  loco::Graph g;

private:
  luci::CircleFullyConnected *fc_node = nullptr;
  luci::CircleInput *input_1 = nullptr;
  luci::CircleInput *input_2 = nullptr;
  luci::CircleConst *weights = nullptr;
  luci::CircleConst *bias = nullptr;
  luci::CircleOutput *output = nullptr;
};

class SimpleInstanceNormGraph
{
public:
  SimpleInstanceNormGraph(bool make_valid)
  {
    instance_norm_node = g.nodes()->create<luci::CircleInstanceNorm>();
    input_1 = g.nodes()->create<luci::CircleInput>();
    gamma = g.nodes()->create<luci::CircleConst>();

    instance_norm_node->input(input_1);
    instance_norm_node->gamma(gamma);

    if (make_valid)
    {
      beta = g.nodes()->create<luci::CircleConst>();
      instance_norm_node->beta(beta);
    }
    else
    {
      input_2 = g.nodes()->create<luci::CircleInput>();
      instance_norm_node->beta(input_2);
    }

    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    output->from(instance_norm_node);
  }

public:
  loco::Graph g;

private:
  luci::CircleInstanceNorm *instance_norm_node = nullptr;
  luci::CircleInput *input_1 = nullptr;
  luci::CircleInput *input_2 = nullptr;
  luci::CircleConst *gamma = nullptr;
  luci::CircleConst *beta = nullptr;
  luci::CircleOutput *output = nullptr;
};

class SimpleTransposeConvGraph
{
public:
  SimpleTransposeConvGraph(bool make_valid)
  {
    transpose_conv = g.nodes()->create<luci::CircleTransposeConv>();
    input_1 = g.nodes()->create<luci::CircleInput>();

    input_sizes = g.nodes()->create<luci::CircleConst>();
    filter = g.nodes()->create<luci::CircleConst>();

    transpose_conv->outBackprop(input_1);
    transpose_conv->filter(filter);
    transpose_conv->inputSizes(input_sizes);
    transpose_conv->fusedActivationFunction(luci::FusedActFunc::NONE);

    if (make_valid)
    {
      bias = g.nodes()->create<luci::CircleConst>();
      transpose_conv->bias(bias);
    }
    else
    {
      input_2 = g.nodes()->create<luci::CircleInput>();
      transpose_conv->bias(input_2);
    }

    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    output->from(transpose_conv);
  }

public:
  loco::Graph g;

private:
  luci::CircleTransposeConv *transpose_conv = nullptr;
  luci::CircleInput *input_1 = nullptr;
  luci::CircleInput *input_2 = nullptr;
  luci::CircleConst *input_sizes = nullptr;
  luci::CircleConst *filter = nullptr;
  luci::CircleConst *bias = nullptr;
  luci::CircleOutput *output = nullptr;
};

class SimplePReluGraph
{
public:
  SimplePReluGraph(bool make_valid)
  {
    prelu = g.nodes()->create<luci::CirclePRelu>();
    input_1 = g.nodes()->create<luci::CircleInput>();

    prelu->input(input_1);

    if (make_valid)
    {
      alpha = g.nodes()->create<luci::CircleConst>();
      prelu->alpha(alpha);
    }
    else
    {
      input_2 = g.nodes()->create<luci::CircleInput>();
      prelu->alpha(input_2);
    }

    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    output->from(prelu);
  }

public:
  loco::Graph g;

private:
  luci::CirclePRelu *prelu = nullptr;
  luci::CircleInput *input_1 = nullptr;
  luci::CircleInput *input_2 = nullptr;
  luci::CircleConst *alpha = nullptr;
  luci::CircleOutput *output = nullptr;
};

TEST(QuantizePreCheckerPassTest, name)
{
  luci::QuantizePreCheckerPass pass{};
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

// Test Conv2d
TEST(QuantizePreCheckerPassTest, conv2d)
{
  SimpleConv2DGraph valid_graph(true);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_NO_THROW(checker.run(&valid_graph.g));
}

TEST(QuantizePreCheckerPassTest, conv2d_NEG)
{
  SimpleConv2DGraph invalid_graph(false);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_ANY_THROW(checker.run(&invalid_graph.g));
}

// Test DepthwiseConv2d
TEST(QuantizePreCheckerPassTest, depthwise_conv2d)
{
  SimpleDepthConv2DGraph valid_graph(true);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_NO_THROW(checker.run(&valid_graph.g));
}

TEST(QuantizePreCheckerPassTest, depthwise_conv2d_NEG)
{
  SimpleDepthConv2DGraph invalid_graph(false);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_ANY_THROW(checker.run(&invalid_graph.g));
}

// Test FullyConnected
TEST(QuantizePreCheckerPassTest, fully_connected)
{
  SimpleFCGraph valid_graph(true);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_NO_THROW(checker.run(&valid_graph.g));
}

TEST(QuantizePreCheckerPassTest, fully_connected_NEG)
{
  SimpleFCGraph invalid_graph(false);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_ANY_THROW(checker.run(&invalid_graph.g));
}

// Test InstanceNorm
TEST(QuantizePreCheckerPassTest, instance_norm)
{
  SimpleInstanceNormGraph valid_graph(true);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_NO_THROW(checker.run(&valid_graph.g));
}

TEST(QuantizePreCheckerPassTest, instance_norm_NEG)
{
  SimpleInstanceNormGraph invalid_graph(false);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_ANY_THROW(checker.run(&invalid_graph.g));
}

// Test TransposeConv
TEST(QuantizePreCheckerPassTest, transpose_conv)
{
  SimpleTransposeConvGraph valid_graph(true);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_NO_THROW(checker.run(&valid_graph.g));
}

TEST(QuantizePreCheckerPassTest, transpose_conv_NEG)
{
  SimpleTransposeConvGraph invalid_graph(false);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_ANY_THROW(checker.run(&invalid_graph.g));
}

// Test PRelu
TEST(QuantizePreCheckerPassTest, prelu)
{
  SimplePReluGraph valid_graph(true);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_NO_THROW(checker.run(&valid_graph.g));
}

TEST(QuantizePreCheckerPassTest, prelu_NEG)
{
  SimplePReluGraph invalid_graph(false);

  luci::QuantizePreCheckerPass checker{};

  EXPECT_ANY_THROW(checker.run(&invalid_graph.g));
}
