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

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

#include <memory>

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

struct SimpleQuantGraph
{
  void init(void);

  loco::Graph g;

  luci::CircleInput *input = nullptr;
  luci::CircleOutput *output = nullptr;
  luci::CircleConv2D *conv2d1 = nullptr;
  luci::CircleConv2D *conv2d2 = nullptr;
  luci::CircleConst *filter = nullptr;
  luci::CircleConst *bias = nullptr;
};

// Have two conv layers named "c1" and "c2".
void SimpleQuantGraph::init()
{
  auto graph_input = g.inputs()->create();
  graph_input->shape({1, 1, 1, 1});
  graph_input->dtype(loco::DataType::FLOAT32);

  auto graph_output = g.outputs()->create();
  graph_output->shape({1, 1, 1, 1});
  graph_output->dtype(loco::DataType::FLOAT32);

  input = g.nodes()->create<luci::CircleInput>();
  input->dtype(loco::DataType::FLOAT32);
  input->shape({1, 1, 1, 1});
  input->shape_status(luci::ShapeStatus::VALID);
  input->index(graph_input->index());

  filter = g.nodes()->create<luci::CircleConst>();
  filter->dtype(loco::DataType::FLOAT32);
  filter->size<loco::DataType::FLOAT32>(1 * 1 * 1 * 1);
  filter->shape({1, 1, 1, 1});
  filter->shape_status(luci::ShapeStatus::VALID);

  bias = g.nodes()->create<luci::CircleConst>();
  bias->dtype(loco::DataType::FLOAT32);
  bias->size<loco::DataType::FLOAT32>(1);
  bias->shape({1});
  bias->shape_status(luci::ShapeStatus::VALID);

  conv2d1 = g.nodes()->create<luci::CircleConv2D>();
  conv2d1->dtype(loco::DataType::FLOAT32);
  conv2d1->fusedActivationFunction(luci::FusedActFunc::NONE);
  conv2d1->input(input);
  conv2d1->filter(filter);
  conv2d1->bias(bias);
  conv2d1->padding(luci::Padding::VALID);
  conv2d1->name("c1");

  conv2d2 = g.nodes()->create<luci::CircleConv2D>();
  conv2d2->dtype(loco::DataType::FLOAT32);
  conv2d2->fusedActivationFunction(luci::FusedActFunc::NONE);
  conv2d2->input(input);
  conv2d2->filter(filter);
  conv2d2->bias(conv2d1);
  conv2d2->padding(luci::Padding::VALID);
  conv2d2->name("c2");

  output = g.nodes()->create<luci::CircleOutput>();
  output->dtype(loco::DataType::FLOAT32);
  output->from(conv2d2);
  output->index(graph_output->index());
}

struct SimpleCircleQuantizer
{
  CircleQuantizer::Options *init();
  void quantize(loco::Graph *g) { cq.quantize(g); }

  luci::CircleQuantizer cq;
};

CircleQuantizer::Options *SimpleCircleQuantizer::init(void)
{
  auto options = cq.options();
  options->enable(Algorithms::QuantizeDequantizeWeights);
  options->param(AlgorithmParameters::Quantize_input_model_dtype, "float32");
  options->param(AlgorithmParameters::Quantize_output_model_dtype, "uint8");
  options->param(AlgorithmParameters::Quantize_granularity, "layer");
  return options;
}

using LayerParam = luci::CircleQuantizer::Options::LayerParam;
using LayerParams = luci::CircleQuantizer::Options::LayerParams;
using LayerParamsSet = luci::CircleQuantizer::Options::LayerParamsSet;

TEST(CircleQuantizerTest, quantize_layer_param_set)
{
  SimpleQuantGraph sqg;
  sqg.init();

  LayerParamsSet lpss;
  {
    LayerParams lps1;
    {
      auto lp1 = std::make_shared<LayerParam>();
      lp1->name = "x1";
      lp1->dtype = "int16";
      lp1->granularity = "channel";
      lps1.emplace_back(lp1);
    }
    lpss.emplace_back(lps1);

    LayerParams lps2;
    {
      auto lp2 = std::make_shared<LayerParam>();
      lp2->name = "c1";
      lp2->dtype = "int16";
      lp2->granularity = "channel";
      lps2.emplace_back(lp2);
    }
    lpss.emplace_back(lps2);
  }

  SimpleCircleQuantizer scq;
  auto options = scq.init();
  options->layer_params_set(lpss);

  EXPECT_NO_THROW(scq.quantize(&sqg.g));
}

TEST(CircleQuantizerTest, invalid_layer_params_NEG)
{
  SimpleQuantGraph sqg;
  sqg.init();

  LayerParamsSet lpss;
  {
    // there is no LayerParam with "c1" nor "c2"
    LayerParams lps1;
    {
      auto lp1 = std::make_shared<LayerParam>();
      lp1->name = "x1";
      lp1->dtype = "int16";
      lp1->granularity = "channel";
      lps1.emplace_back(lp1);
    }
    lpss.emplace_back(lps1);

    LayerParams lps2;
    {
      auto lp2 = std::make_shared<LayerParam>();
      lp2->name = "x2";
      lp2->dtype = "int16";
      lp2->granularity = "channel";
      lps2.emplace_back(lp2);
    }
    lpss.emplace_back(lps2);
  }

  SimpleCircleQuantizer scq;
  auto options = scq.init();
  options->layer_params_set(lpss);

  EXPECT_THROW(scq.quantize(&sqg.g), std::runtime_error);
}

TEST(CircleQuantizerTest, duplicate_name_in_layer_params_NEG)
{
  SimpleQuantGraph sqg;
  sqg.init();

  LayerParamsSet lpss;
  {
    LayerParams lps1;
    {
      // duplicate c1 name in a LayerParams
      auto lp11 = std::make_shared<LayerParam>();
      lp11->name = "c1";
      lp11->dtype = "int16";
      lp11->granularity = "channel";
      lps1.emplace_back(lp11);

      auto lp12 = std::make_shared<LayerParam>();
      lp12->name = "c1";
      lp12->dtype = "int16";
      lp12->granularity = "channel";
      lps1.emplace_back(lp12);
    }
    lpss.emplace_back(lps1);

    LayerParams lps2;
    {
      auto lp2 = std::make_shared<LayerParam>();
      lp2->name = "x1";
      lp2->dtype = "int16";
      lp2->granularity = "channel";
      lps2.emplace_back(lp2);
    }
    lpss.emplace_back(lps2);
  }

  SimpleCircleQuantizer scq;
  auto options = scq.init();
  options->layer_params_set(lpss);

  EXPECT_THROW(scq.quantize(&sqg.g), std::runtime_error);
}

TEST(CircleQuantizerTest, quantize_layer_param_set_dup2_NEG)
{
  SimpleQuantGraph sqg;
  sqg.init();

  LayerParamsSet lpss;
  {
    // duplicate "c1" name in a LayerParamsSet
    LayerParams lps1;
    {
      auto lp1 = std::make_shared<LayerParam>();
      lp1->name = "c1";
      lp1->dtype = "int16";
      lp1->granularity = "channel";
      lps1.emplace_back(lp1);
    }
    lpss.emplace_back(lps1);

    LayerParams lps2;
    {
      auto lp2 = std::make_shared<LayerParam>();
      lp2->name = "c1";
      lp2->dtype = "int16";
      lp2->granularity = "channel";
      lps2.emplace_back(lp2);
    }
    lpss.emplace_back(lps2);
  }

  SimpleCircleQuantizer scq;
  auto options = scq.init();
  options->layer_params_set(lpss);

  EXPECT_THROW(scq.quantize(&sqg.g), std::runtime_error);
}
