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

#include "Quantizer.h"
#include <luci/Service/Validate.h>

#include <iostream>

using namespace mpqsolver;
using AlgorithmParameters = luci::CircleQuantizer::Options::AlgorithmParameters;
using Algorithms = luci::CircleQuantizer::Options::Algorithm;

namespace mpqsolver
{

bool make_model_fake_quantized(luci::Module *module)
{
  luci::CircleQuantizer quantizer;

  auto options = quantizer.options();
  options->enable(Algorithms::ConvertToFakeQuantizedModel);

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);
    // quantize the graph
    quantizer.quantize(graph);
    if (!luci::validate(graph))
    {
      return false;
    }
  }

  return true;
}

} // namespace mpqsolver

bool Quantizer::quantize(luci::Module *module, const std::string &def_quant,
                         LayerParams &layer_params)
{
  if (!module)
    return false;

  static const std::string default_dtype = "float32";
  static const std::string input_dtype = "uint8";
  static const std::string output_dtype = "uint8";
  static const std::string granularity_type = "channel";

  luci::CircleQuantizer quantizer;

  auto options = quantizer.options();
  options->enable(Algorithms::QuantizeWithMinMax);

  options->param(AlgorithmParameters::Quantize_input_model_dtype, default_dtype);
  options->param(AlgorithmParameters::Quantize_output_model_dtype, def_quant);
  options->param(AlgorithmParameters::Quantize_granularity, granularity_type);
  options->param(AlgorithmParameters::Quantize_input_type, input_dtype);
  options->param(AlgorithmParameters::Quantize_output_type, output_dtype);
  options->param(AlgorithmParameters::Quantize_TF_style_maxpool, "True");

  if (!layer_params.empty())
  {
    try
    {
      options->layer_params(AlgorithmParameters::Quantize_layer_params, layer_params);
    }
    catch (const std::runtime_error &e)
    {
      std::cerr << e.what() << '\n';
      return false;
    }
  }

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);
    // quantize the graph
    quantizer.quantize(graph);
    if (!luci::validate(graph))
    {
      std::cerr << "ERROR: Quantized graph is invalid" << std::endl;
      return false;
    }
  }

  return true;
}

bool Quantizer::fake_quantize(luci::Module *module, const std::string &def_quant,
                              LayerParams &layer_params)
{
  if (!quantize(module, def_quant, layer_params))
    return false;

  if (!make_model_fake_quantized(module))
    return false;

  return true;
}
