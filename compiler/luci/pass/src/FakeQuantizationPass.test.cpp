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

#include <logo/Phase.h>

#include "luci/Pass/FakeQuantizationPass.h"
#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

// Check the below pattern
// Quantize (scale, zp) -> Dequantize (node)
void check_q_dq(loco::Node *node, float scale, int64_t zp)
{
  auto dequant = dynamic_cast<luci::CircleDequantize *>(node);
  EXPECT_TRUE(dequant != nullptr);
  auto quant = dynamic_cast<luci::CircleQuantize *>(dequant->input());
  EXPECT_TRUE(quant != nullptr);
  auto qparam = quant->quantparam();
  EXPECT_EQ(scale, qparam->scale[0]);
  EXPECT_EQ(zp, qparam->zerop[0]);
}

// Check the below pattern
// Dequantize (node)
void check_dq(loco::Node *node)
{
  auto dequant = dynamic_cast<luci::CircleDequantize *>(node);
  EXPECT_TRUE(dequant != nullptr);
}

void set_qparam(luci::CircleNode *node, float scale, int64_t zp)
{
  auto qparam = std::make_unique<luci::CircleQuantParam>();
  {
    qparam->scale.push_back(scale);
    qparam->zerop.push_back(zp);
  }
  node->quantparam(std::move(qparam));
}

/**
 *  SimpleGraph for testing
 *  - Child class should implement insertGraphBody()
 *
 *  Example (U8ConvGraph inherits SimpleGraph and create Conv2D Op)
 *
 *  BEFORE
 *  - A model is quantized (ex: u8)
 *
 *  [Input(u8)] [Filter(u8)] [Bias(s32)]
 *           \       |        /
 *            \      |       /
 *             \     |      /
 *              [Conv2D(u8)]
 *                   |
 *              [Output(u8)]
 *
 *  AFTER
 *  - Ops are converted to fp32
 *  - Quantize/Dequantize Ops are inserted properly
 *    - Q-DQ is inserted after non-const activation
 *    - DQ is inserted after const
 *
 *  [Input(u8)]
 *        |
 *  [Quant(u8)]     [Filter(u8)]       [Bias(s32)]
 *        |              |                 |
 *  [Dequant(fp32)] [Dequant(fp32)] [Dequant(fp32)]
 *             \         |          /
 *              \        |         /
 *               \       |        /
 *                 [Conv2D(fp32)]
 *                       |
 *                  [Quant(u8)]
 *                       |
 *                 [Dequant(fp32)]
 *                       |
 *                  [Output(fp32)]
 */
template <loco::DataType T> class SimpleGraph
{
public:
  void init()
  {
    input = g.nodes()->create<luci::CircleInput>();
    output = g.nodes()->create<luci::CircleOutput>();
    input->name("input");
    output->name("output");

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    graph_input->dtype(T);
    input->dtype(T);
    output->dtype(T);
    graph_output->dtype(T);

    graph_input->shape({1, 4, 4, 4});
    input->shape({1, 4, 4, 4});
    output->shape({1, 4, 4, 4});
    graph_output->shape({1, 4, 4, 4});

    set_qparam(input, 1.0, 0);
    set_qparam(output, 1.0, 0);

    auto graph_body = insertGraphBody(input);
    output->from(graph_body);
  }

  virtual ~SimpleGraph() = default;

protected:
  virtual loco::Node *insertGraphBody(loco::Node *input) = 0;

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleOutput *output = nullptr;
};

class U8ConvGraph final : public SimpleGraph<loco::DataType::U8>
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    conv = g.nodes()->create<luci::CircleConv2D>();
    weights = g.nodes()->create<luci::CircleConst>();
    bias = g.nodes()->create<luci::CircleConst>();

    conv->dtype(loco::DataType::U8);
    weights->dtype(loco::DataType::U8);
    bias->dtype(loco::DataType::S32);

    conv->shape({1, 4, 4, 4});
    weights->shape({4, 1, 1, 4});
    bias->shape({4});

    weights->size<loco::DataType::U8>(16);
    for (uint32_t i = 0; i < 16; i++)
      weights->at<loco::DataType::U8>(i) = i;

    bias->size<loco::DataType::S32>(4);
    for (uint32_t i = 0; i < 4; i++)
      bias->at<loco::DataType::S32>(i) = i;

    set_qparam(conv, 2.0, 127);
    set_qparam(weights, 2.0, 127);
    set_qparam(bias, 2.0, 127);

    conv->input(input);
    conv->filter(weights);
    conv->bias(bias);

    conv->name("conv");
    weights->name("weights");
    bias->name("bias");

    return conv;
  }

public:
  luci::CircleConv2D *conv = nullptr;
  luci::CircleConst *weights = nullptr;
  luci::CircleConst *bias = nullptr;
};

class FP32ConvGraph final : public SimpleGraph<loco::DataType::FLOAT32>
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    conv = g.nodes()->create<luci::CircleConv2D>();
    weights = g.nodes()->create<luci::CircleConst>();
    bias = g.nodes()->create<luci::CircleConst>();

    conv->dtype(loco::DataType::FLOAT32);
    weights->dtype(loco::DataType::FLOAT32);
    bias->dtype(loco::DataType::FLOAT32);

    conv->shape({1, 4, 4, 4});
    weights->shape({4, 1, 1, 4});
    bias->shape({4});

    weights->size<loco::DataType::FLOAT32>(16);
    for (uint32_t i = 0; i < 16; i++)
      weights->at<loco::DataType::FLOAT32>(i) = i;

    bias->size<loco::DataType::FLOAT32>(4);
    for (uint32_t i = 0; i < 4; i++)
      bias->at<loco::DataType::FLOAT32>(i) = i;

    conv->input(input);
    conv->filter(weights);
    conv->bias(bias);

    conv->name("conv");
    weights->name("weights");
    bias->name("bias");

    return conv;
  }

public:
  luci::CircleConv2D *conv = nullptr;
  luci::CircleConst *weights = nullptr;
  luci::CircleConst *bias = nullptr;
};

} // namespace

TEST(FakeQuantization, U8Conv2D)
{
  U8ConvGraph g;
  g.init();

  luci::FakeQuantizationPass fq;
  fq.run(&g.g);

  // Check ifm
  check_q_dq(g.conv->input(), 1.0, 0);

  // Check weights
  check_dq(g.conv->filter());

  // Check bias
  check_dq(g.conv->bias());

  // Check ofm
  check_q_dq(g.output->from(), 2.0, 127);

  SUCCEED();
}

TEST(FakeQuantization, F32Conv2D_NEG)
{
  FP32ConvGraph g;
  g.init();

  luci::FakeQuantizationPass fq;
  fq.run(&g.g);

  uint32_t dequant_count = 0;
  uint32_t quant_count = 0;

  for (auto node : loco::active_nodes(loco::output_nodes(&g.g)))
  {
    auto cnode = loco::must_cast<luci::CircleNode *>(node);
    auto opcode = cnode->opcode();
    if (opcode == luci::CircleOpcode::DEQUANTIZE)
      dequant_count++;
    if (opcode == luci::CircleOpcode::QUANTIZE)
      quant_count++;
  }

  // Check no quant/dequant Op is inserted
  EXPECT_EQ(0, quant_count);
  EXPECT_EQ(0, dequant_count);
}
