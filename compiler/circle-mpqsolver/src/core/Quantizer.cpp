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

using namespace mpqsolver::core;
using AlgorithmParameters = luci::CircleQuantizer::Options::AlgorithmParameters;
using Algorithms = luci::CircleQuantizer::Options::Algorithm;

namespace
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

} // namespace

void Quantizer::set_hook(const QuantizerHook *hook) { _hook = hook; }

/**
 * @brief quantize recorded module (min/max initialized) with specified parameters
 * returns true on success
 */
bool Quantizer::quantize(luci::Module *module, const std::string &quant_dtype,
                         LayerParams &layer_params)
{
  if (!module)
    return false;

  static const std::string default_dtype = "float32";

  luci::CircleQuantizer quantizer;

  auto options = quantizer.options();
  options->enable(Algorithms::QuantizeWithMinMax);

  options->param(AlgorithmParameters::Quantize_input_model_dtype, default_dtype);
  options->param(AlgorithmParameters::Quantize_output_model_dtype, quant_dtype);
  // Only channel-wise quantization is supported for int16
  // TODO Fix this if this assumption breaks
  if (quant_dtype == "int16")
    options->param(AlgorithmParameters::Quantize_granularity, "channel");
  else
    options->param(AlgorithmParameters::Quantize_granularity, _ctx.granularity);

  options->param(AlgorithmParameters::Quantize_input_type, _ctx.input_type);
  options->param(AlgorithmParameters::Quantize_output_type, _ctx.output_type);
  options->param(AlgorithmParameters::Quantize_TF_style_maxpool,
                 _ctx.TF_style_maxpool ? "True" : "False");
  options->param(AlgorithmParameters::Quantize_save_min_max, _ctx.save_min_max ? "True" : "False");

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

  if (_hook)
  {
    _hook->on_quantized(module);
  }

  return true;
}

/**
 * @brief quantize recorded module (min/max initialized) with specified parameters
 * returns true on success
 */

bool Quantizer::quantize(luci::Module *module, LayerParams &layer_params)
{
  return quantize(module, _ctx.output_model_dtype, layer_params);
}

/**
 * @brief fake_quantize recorded module (min/max initialized) with specified parameters
 * returns true on success
 */
bool Quantizer::fake_quantize(luci::Module *module, const std::string &quant_dtype,
                              LayerParams &layer_params)
{
  if (!quantize(module, quant_dtype, layer_params))
    return false;

  if (!make_model_fake_quantized(module))
    return false;

  return true;
}
