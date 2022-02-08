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

#include "luci/CircleQuantizer.h"

#include <gtest/gtest.h>

using namespace luci;
using Algorithms = luci::CircleQuantizer::Options::Algorithm;
using AlgorithmParameters = luci::CircleQuantizer::Options::AlgorithmParameters;

TEST(CircleQuantizerTest, quantize_quantdequant_simple)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::QuantizeDequantizeWeights);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "float32");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "uint8");
  options->param(AlgorithmParameters::Quantize_granularity, "layer");

  o.quantize(&g);

  SUCCEED();
}

TEST(CircleQuantizerTest, quantize_quantdequant_input_NEG)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::QuantizeDequantizeWeights);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "invalid");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "uint8");
  options->param(AlgorithmParameters::Quantize_granularity, "layer");

  EXPECT_THROW(o.quantize(&g), std::runtime_error);
}

TEST(CircleQuantizerTest, quantize_quantdequant_output_NEG)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::QuantizeDequantizeWeights);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "float32");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "invalid");
  options->param(AlgorithmParameters::Quantize_granularity, "layer");

  EXPECT_THROW(o.quantize(&g), std::runtime_error);
}

TEST(CircleQuantizerTest, quantize_quantdequant_gran_NEG)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::QuantizeDequantizeWeights);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "float32");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "uint8");
  options->param(AlgorithmParameters::Quantize_granularity, "invalid");

  EXPECT_THROW(o.quantize(&g), std::runtime_error);
}

TEST(CircleQuantizerTest, quantize_minmax_simple)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::QuantizeWithMinMax);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "float32");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "uint8");
  options->param(AlgorithmParameters::Quantize_granularity, "layer");

  o.quantize(&g);

  SUCCEED();
}

TEST(CircleQuantizerTest, quantize_minmax_input_NEG)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::QuantizeWithMinMax);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "invalid");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "uint8");
  options->param(AlgorithmParameters::Quantize_granularity, "layer");

  EXPECT_THROW(o.quantize(&g), std::runtime_error);
}

TEST(CircleQuantizerTest, quantize_minmax_output_NEG)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::QuantizeWithMinMax);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "float32");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "invalid");
  options->param(AlgorithmParameters::Quantize_granularity, "layer");

  EXPECT_THROW(o.quantize(&g), std::runtime_error);
}

TEST(CircleQuantizerTest, quantize_minmax_gran_NEG)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::QuantizeWithMinMax);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "float32");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "uint8");
  options->param(AlgorithmParameters::Quantize_granularity, "invalid");

  EXPECT_THROW(o.quantize(&g), std::runtime_error);
}

TEST(CircleQuantizerTest, quantize_requant_simple)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::Requantize);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "int8");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "uint8");

  o.quantize(&g);

  SUCCEED();
}

TEST(CircleQuantizerTest, quantize_requant_input_NEG)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::Requantize);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "invalid");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "uint8");

  EXPECT_THROW(o.quantize(&g), std::runtime_error);
}

TEST(CircleQuantizerTest, quantize_requant_output_NEG)
{
  loco::Graph g;
  luci::CircleQuantizer o;

  auto options = o.options();

  options->enable(Algorithms::Requantize);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "int8");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "invalid");

  EXPECT_THROW(o.quantize(&g), std::runtime_error);
}
