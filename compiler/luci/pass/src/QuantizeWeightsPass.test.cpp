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

#include "luci/Pass/QuantizeWeightsPass.h"
#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{
struct QuantizeWeightsPassTest : public ::testing::Test
{
  /**
   *  nconv graph
   *
   *        [CircleInput]
   *              |
   *              |
   *        [CircleConv2D]
   *              |
   *              |
   *        [CircleOutput]
   */
  void MakeGraph()
  {
    const int N = 1;
    const int H = 4;
    const int W = 4;
    const int C = 3; // IC = OC

    // graph input and output
    auto graph_input = _g.inputs()->create();
    auto graph_output = _g.outputs()->create();

    // CircleInput
    auto input = _g.nodes()->create<luci::CircleInput>();
    input->index(graph_input->index());
    input->shape({N, H, W, C});
    input->dtype(loco::DataType::FLOAT32);
    input->name("input");

    // CircleConv2D
    auto conv = _g.nodes()->create<luci::CircleConv2D>();
    conv->input(input);
    auto bias = _g.nodes()->create<luci::CircleConst>();
    bias->dtype(loco::DataType::FLOAT32);
    bias->shape({C});
    bias->name("conv_bias");
    conv->bias(bias);
    auto weight = _g.nodes()->create<luci::CircleConst>();
    weight->dtype(loco::DataType::FLOAT32);
    weight->shape({C, H, W, C});
    weight->size<loco::DataType::FLOAT32>(C * H * W * C);
    conv->filter(weight);
    conv->padding(luci::Padding::SAME);
    conv->fusedActivationFunction(luci::FusedActFunc::NONE);
    conv->dtype(loco::DataType::FLOAT32);
    conv->name("nconv");

    // CircleOutput
    auto output = _g.nodes()->create<luci::CircleOutput>();
    output->index(graph_output->index());
    output->from(conv);
    output->shape({N, H, W, C});
    output->dtype(loco::DataType::FLOAT32);
    output->name("output");
  }
  virtual void SetUp() { MakeGraph(); }
  loco::Graph _g;
};

} // namespace

TEST_F(QuantizeWeightsPassTest, name)
{
  luci::QuantizeWeightsPass pass(loco::DataType::FLOAT32, loco::DataType::S8,
                                 luci::QuantizationGranularity::ChannelWise);
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(QuantizeWeightsPassTest, name_ctx)
{
  auto ctx = std::make_unique<luci::QuantizeWeightsPass::Context>();
  {
    ctx->input_model_dtype = loco::DataType::FLOAT32;
    ctx->output_model_dtype = loco::DataType::S8;
    ctx->granularity = luci::QuantizationGranularity::ChannelWise;
  }

  luci::QuantizeWeightsPass pass(std::move(ctx));
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(QuantizeWeightsPassTest, run_input_U8_NEG)
{
  loco::Graph g;
  luci::QuantizeWeightsPass pass(loco::DataType::U8, loco::DataType::S8,
                                 luci::QuantizationGranularity::ChannelWise);
  EXPECT_THROW(pass.run(&_g), std::runtime_error);
}

TEST_F(QuantizeWeightsPassTest, run_output_f32_NEG)
{
  loco::Graph g;
  luci::QuantizeWeightsPass pass(loco::DataType::FLOAT32, loco::DataType::FLOAT32,
                                 luci::QuantizationGranularity::ChannelWise);
  EXPECT_THROW(pass.run(&_g), std::runtime_error);
}
