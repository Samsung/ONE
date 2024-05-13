/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "QuantizeWithPredecessorPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

void addQuantParam(luci::CircleNode *node, const std::vector<float> &scale,
                   const std::vector<int64_t> &zp)
{
  assert(node->quantparam() == nullptr);

  auto quantparam = std::make_unique<luci::CircleQuantParam>();
  quantparam->scale = scale;
  quantparam->zerop = zp;
  node->quantparam(std::move(quantparam));
}

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 *        [Conv] (int16)
 *           |
 *       [Reshape] (fp32)
 *
 *  AFTER
 *
 *        [Conv] (int16)
 *           |
 *       [Reshape] (int16)
 *
 */
class ConvReshapeGraph
{
public:
  ConvReshapeGraph()
  {
    input = g.nodes()->create<luci::CircleInput>();
    conv = g.nodes()->create<luci::CircleConv2D>();
    reshape = g.nodes()->create<luci::CircleReshape>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    conv->dtype(loco::DataType::S16);
    reshape->dtype(loco::DataType::FLOAT32);

    addQuantParam(conv, {1.0}, {0});

    conv->input(input);
    reshape->tensor(conv);
    output->from(reshape);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleConv2D *conv = nullptr;
  luci::CircleReshape *reshape = nullptr;
  luci::CircleOutput *output = nullptr;
};

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 *        [Conv] (int16)
 *           |
 *       [Squeeze] (fp32)
 *
 *  AFTER
 *
 *        [Conv] (int16)
 *           |
 *       [Squeeze] (int16)
 *
 */
class ConvSqueezeGraph
{
public:
  ConvSqueezeGraph()
  {
    input = g.nodes()->create<luci::CircleInput>();
    conv = g.nodes()->create<luci::CircleConv2D>();
    squeeze = g.nodes()->create<luci::CircleSqueeze>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    conv->dtype(loco::DataType::S16);
    squeeze->dtype(loco::DataType::FLOAT32);

    addQuantParam(conv, {1.0}, {0});

    squeeze->squeeze_dims({0});

    conv->input(input);
    squeeze->input(conv);
    output->from(squeeze);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleConv2D *conv = nullptr;
  luci::CircleSqueeze *squeeze = nullptr;
  luci::CircleOutput *output = nullptr;
};

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 *        [Conv] (int16)
 *           |
 *         [Mul] (fp32)
 *
 *  AFTER
 *
 *        [Conv] (int16)
 *           |
 *         [Mul] (int16)
 *
 */
class ConvMulGraph
{
public:
  ConvMulGraph()
  {
    input = g.nodes()->create<luci::CircleInput>();
    conv = g.nodes()->create<luci::CircleConv2D>();
    mul = g.nodes()->create<luci::CircleMul>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    conv->dtype(loco::DataType::S16);
    mul->dtype(loco::DataType::FLOAT32);

    addQuantParam(conv, {1.0}, {0});

    conv->input(input);
    mul->x(conv);
    mul->y(conv);
    output->from(mul);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleConv2D *conv = nullptr;
  luci::CircleMul *mul = nullptr;
  luci::CircleOutput *output = nullptr;
};

} // namespace

TEST(QuantizeWithPredecessor, reshape)
{
  ConvReshapeGraph g;

  luci::QuantizeWithPredecessorPass pass;
  while (pass.run(&g.g))
    ;

  EXPECT_NE(nullptr, g.reshape->quantparam());
  EXPECT_FLOAT_EQ(1.0, g.reshape->quantparam()->scale[0]);
  EXPECT_EQ(0, g.reshape->quantparam()->zerop[0]);
}

TEST(QuantizeWithPredecessor, reshape_NEG)
{
  ConvReshapeGraph g;
  g.conv->quantparam(nullptr);

  luci::QuantizeWithPredecessorPass pass;
  EXPECT_FALSE(pass.run(&g.g));
}

TEST(QuantizeWithPredecessor, squeeze)
{
  ConvSqueezeGraph g;

  luci::QuantizeWithPredecessorPass pass;
  while (pass.run(&g.g))
    ;

  EXPECT_NE(nullptr, g.squeeze->quantparam());
  EXPECT_FLOAT_EQ(1.0, g.squeeze->quantparam()->scale[0]);
  EXPECT_EQ(0, g.squeeze->quantparam()->zerop[0]);
}

TEST(QuantizeWithPredecessor, squeeze_NEG)
{
  ConvSqueezeGraph g;
  g.conv->quantparam(nullptr);

  luci::QuantizeWithPredecessorPass pass;
  EXPECT_FALSE(pass.run(&g.g));
}

TEST(QuantizeWithPredecessor, mul)
{
  ConvMulGraph g;

  luci::QuantizeWithPredecessorPass pass;
  while (pass.run(&g.g))
    ;

  EXPECT_NE(nullptr, g.mul->quantparam());
  EXPECT_FLOAT_EQ(32767, g.mul->quantparam()->scale[0]);
  EXPECT_EQ(0, g.mul->quantparam()->zerop[0]);
}

TEST(QuantizeWithPredecessor, mul_NEG)
{
  ConvMulGraph g;
  g.conv->quantparam(nullptr);

  luci::QuantizeWithPredecessorPass pass;
  EXPECT_FALSE(pass.run(&g.g));
}
