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

#include "BisectionSolver.h"
#include "DepthParameterizer.h"
#include "ErrorMetric.h"

#include <luci/ImporterEx.h>
#include <luci/Log.h>

#include <cmath>
#include <iostream>

using namespace mpqsolver::bisection;

namespace
{

std::unique_ptr<luci::Module> read_module(const std::string &path)
{
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(path);
  if (module.get() == nullptr)
  {
    std::cerr << "ERROR: Failed to load " << path << std::endl;
    return nullptr;
  }

  return module;
}

} // namespace

BisectionSolver::BisectionSolver(const std::string &input_data_path, float qerror_ratio,
                                 const std::string &input_quantization,
                                 const std::string &output_quantization)
  : MPQSolver(input_data_path, qerror_ratio, input_quantization, output_quantization)
{
  _quantizer = std::make_unique<Quantizer>(_input_quantization, _output_quantization);
}

float BisectionSolver::evaluate(const DatasetEvaluator &evaluator, const std::string &flt_path,
                                const std::string &def_quant, LayerParams &layers)
{
  auto model = read_module(flt_path);
  // get fake quantized model for evaluation
  if (!_quantizer->fake_quantize(model.get(), def_quant, layers))
  {
    throw std::runtime_error("Failed to produce fake-quantized model.");
  }

  return evaluator.evaluate(model.get());
}

void BisectionSolver::algorithm(Algorithm algorithm) { _algorithm = algorithm; }

std::unique_ptr<luci::Module> BisectionSolver::run(const std::string &module_path)
{
  LOGGER(l);

  auto module = read_module(module_path);

  float min_depth = 0.f;
  float max_depth = 0.f;
  NodeDepthType nodes_depth;
  if (compute_depth(module.get(), nodes_depth, min_depth, max_depth) !=
      ParameterizerResult::SUCCESS)
  {
    std::cerr << "ERROR: Invalid graph for bisectioning" << std::endl;
    return nullptr;
  }

  std::unique_ptr<MAEMetric> metric = std::make_unique<MAEMetric>();
  DatasetEvaluator evaluator(module.get(), _input_data_path, *metric.get());

  LayerParams layer_params;
  float int16_qerror =
    evaluate(evaluator, module_path, "int16" /* default quant_dtype */, layer_params);
  VERBOSE(l, 0) << "Full int16 model quantization error " << int16_qerror << std::endl;

  float uint8_qerror =
    evaluate(evaluator, module_path, "uint8" /* default quant_dtype */, layer_params);
  VERBOSE(l, 0) << "Full uint8 model quantization error " << uint8_qerror << std::endl;

  if (int16_qerror > uint8_qerror)
  {
    throw std::runtime_error("Q8 model's qerror is less than Q16 model's qerror.");
  }

  _qerror = int16_qerror + _qerror_ratio * std::fabs(uint8_qerror - int16_qerror);
  VERBOSE(l, 0) << "Target quantization error " << _qerror << std::endl;

  if (uint8_qerror <= _qerror)
  {
    // no need for bisectioning just return Q8 model
    if (!_quantizer->quantize(module.get(), "uint8", layer_params))
    {
      std::cerr << "ERROR: Failed to quantize model" << std::endl;
      return nullptr;
    }
  }

  int last_depth = -1;
  float best_depth = -1;
  LayerParams best_params;
  if (module->size() != 1)
  {
    throw std::runtime_error("Unsupported module");
  }
  auto graph = module->graph(0);
  auto active_nodes = loco::active_nodes(loco::output_nodes(graph));
  // input and output nodes are not valid for quantization, so let's remove them
  for (auto node : loco::input_nodes(graph))
  {
    active_nodes.erase(node);
  }
  for (auto node : loco::output_nodes(graph))
  {
    active_nodes.erase(node);
  }

  // let's decide whether nodes at input are more suspectible to be quantized into Q16, than at
  // output
  bool int16_front = true;
  switch (_algorithm)
  {
    case Algorithm::Auto: // TODO
    case Algorithm::ForceQ16Front:
      int16_front = true;
      break;
    case Algorithm::ForceQ16Back:
      int16_front = true;
      break;
  }

  while (true)
  {
    int cut_depth = static_cast<int>(std::floor(0.5f * (min_depth + max_depth)));

    if (last_depth == cut_depth)
    {
      break;
    }
    last_depth = cut_depth;

    LayerParams layer_params;
    for (auto &node : active_nodes)
    {
      auto cur_node = loco::must_cast<luci::CircleNode *>(node);
      auto iter = nodes_depth.find(cur_node);
      if (iter == nodes_depth.end())
      {
        continue; // to filter out nodes like weights
      }

      float depth = iter->second;

      if ((depth <= cut_depth && int16_front) || (depth >= cut_depth && !int16_front))
      {
        auto layer_param = std::make_shared<LayerParam>();
        {
          layer_param->name = cur_node->name();
          layer_param->dtype = "int16";
          layer_param->granularity = "channel";
        }

        layer_params.emplace_back(layer_param);
      }
    }

    float cur_accuracy = evaluate(evaluator, module_path, "uint8", layer_params);
    VERBOSE(l, 0) << cut_depth << " : " << cur_accuracy << std::endl;

    if (cur_accuracy < _qerror)
    {
      int16_front ? (max_depth = cut_depth) : (min_depth = cut_depth);
      best_params = layer_params;
      best_depth = cut_depth;
    }
    else
    {
      int16_front ? (min_depth = cut_depth) : (max_depth = cut_depth);
    }
  }

  VERBOSE(l, 0) << "Found the best configuration at " << best_depth << " depth." << std::endl;
  if (!_quantizer->quantize(module.get(), "uint8", best_params))
  {
    std::cerr << "ERROR: Failed to quantize model" << std::endl;
    return nullptr;
  }

  return module;
}
