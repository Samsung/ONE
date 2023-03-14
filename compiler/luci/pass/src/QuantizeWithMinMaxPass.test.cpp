/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/QuantizeWithMinMaxPass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

class SimpleConcatGraph
{
public:
  SimpleConcatGraph(loco::DataType quant_type)
  {
    concat_node = g.nodes()->create<luci::CircleConcatenation>(2);
    input_1 = g.nodes()->create<luci::CircleConst>();
    input_2 = g.nodes()->create<luci::CircleConst>();

    concat_node->dtype(quant_type);
    concat_node->fusedActivationFunction(luci::FusedActFunc::NONE);
    input_1->dtype(quant_type);
    input_2->dtype(quant_type);

    concat_node->values(0, input_1);
    concat_node->values(1, input_2);
  }

  ~SimpleConcatGraph()
  {
    concat_node->values(0, nullptr);
    concat_node->values(1, nullptr);
  }

public:
  loco::Graph g;
  luci::CircleConcatenation *concat_node = nullptr;
  luci::CircleConst *input_1 = nullptr;
  luci::CircleConst *input_2 = nullptr;
};

TEST(QuantizeWithMinMaxPassTest, name)
{
  auto ctx = std::make_unique<luci::QuantizeWithMinMaxPass::Context>();
  {
    ctx->input_model_dtype = loco::DataType::FLOAT32;
    ctx->output_model_dtype = loco::DataType::U8;
    ctx->granularity = luci::QuantizationGranularity::LayerWise;
  }

  luci::QuantizeWithMinMaxPass pass(std::move(ctx));
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

// Test concat of integer tensors
// Integer tensors are not quantized
TEST(QuantizeWithMinMaxPassTest, int_concat)
{
  SimpleConcatGraph g(loco::DataType::S32);

  auto ctx = std::make_unique<luci::QuantizeWithMinMaxPass::Context>();
  {
    ctx->input_model_dtype = loco::DataType::FLOAT32;
    ctx->output_model_dtype = loco::DataType::U8;
    ctx->granularity = luci::QuantizationGranularity::LayerWise;
  }

  luci::QuantizeWithMinMaxPass qwmm(std::move(ctx));

  qwmm.run(&g.g);

  EXPECT_EQ(nullptr, g.concat_node->quantparam());
  EXPECT_EQ(nullptr, g.input_1->quantparam());
  EXPECT_EQ(nullptr, g.input_2->quantparam());
}

TEST(QuantizeWithMinMaxPassTest, inactive_input)
{
  SimpleConcatGraph g(loco::DataType::FLOAT32);

  // Unused input
  g.g.nodes()->create<luci::CircleInput>();

  auto ctx = std::make_unique<luci::QuantizeWithMinMaxPass::Context>();
  {
    ctx->input_model_dtype = loco::DataType::FLOAT32;
    ctx->output_model_dtype = loco::DataType::U8;
    ctx->granularity = luci::QuantizationGranularity::LayerWise;
  }

  luci::QuantizeWithMinMaxPass qwmm(std::move(ctx));

  EXPECT_NO_THROW(qwmm.run(&g.g));
}
