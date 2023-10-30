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
#include "VISQErrorApproximator.h"

#include "core/ErrorMetric.h"
#include "core/SolverOutput.h"

#include <luci/ImporterEx.h>

#include <cmath>
#include <iostream>

using namespace mpqsolver::bisection;

namespace
{

/**
 * @brief Compare errors of two disjoint subsets of a model sliced by cut_depth
 * @return True if the front part (< cut_depth) has larger errors than the rear part (>= cut_depth)
 */
bool front_has_higher_error(const NodeDepthType &nodes_depth, const std::string &visq_path,
                            float cut_depth)
{
  SolverOutput::get() << "\n>> Running bisection(auto) algorithm\n";

  VISQErrorApproximator approximator;
  approximator.init(visq_path);

  float error_at_input = 0;
  float error_at_output = 0;
  for (auto &iter : nodes_depth)
  {
    float cur_error = approximator.approximate(iter.first->name());
    if (iter.second < cut_depth)
    {
      error_at_input += cur_error;
    }
    else
    {
      error_at_output += cur_error;
    }
  }

  SolverOutput::get() << "Qerror of front half: " << error_at_input << "\n";
  SolverOutput::get() << "Qerror of rear half: " << error_at_output << "\n";
  if (error_at_input > error_at_output)
  {
    SolverOutput::get() << "Front part will be Q16, while the rear will be Q8\n";
  }
  else
  {
    SolverOutput::get() << "Front part will be Q8, while the rear will be Q16\n";
  }

  return error_at_input > error_at_output;
}

} // namespace

BisectionSolver::BisectionSolver(const std::string &input_data_path, float qerror_ratio,
                                 const std::string &input_quantization,
                                 const std::string &output_quantization)
  : MPQSolver(input_data_path, qerror_ratio, input_quantization, output_quantization)
{
}

float BisectionSolver::evaluate(const core::DatasetEvaluator &evaluator,
                                const std::string &flt_path, const std::string &def_quant,
                                core::LayerParams &layers)
{
  auto model = read_module(flt_path);
  assert(model != nullptr);

  // get fake quantized model for evaluation
  if (!_quantizer->fake_quantize(model.get(), def_quant, layers))
  {
    throw std::runtime_error("Failed to produce fake-quantized model.");
  }

  return evaluator.evaluate(model.get());
}

void BisectionSolver::algorithm(Algorithm algorithm) { _algorithm = algorithm; }

void BisectionSolver::setVisqPath(const std::string &visq_path) { _visq_data_path = visq_path; }

std::unique_ptr<luci::Module> BisectionSolver::run(const std::string &module_path)
{
  auto module = read_module(module_path);
  assert(module != nullptr);

  float min_depth = 0.f;
  float max_depth = 0.f;
  NodeDepthType nodes_depth;
  if (compute_depth(module.get(), nodes_depth, min_depth, max_depth) !=
      ParameterizerResult::SUCCESS)
  {
    std::cerr << "ERROR: Invalid graph for bisectioning" << std::endl;
    return nullptr;
  }

  SolverOutput::get() << "\n>> Computing baseline qerrors\n";

  std::unique_ptr<core::MAEMetric> metric = std::make_unique<core::MAEMetric>();
  core::DatasetEvaluator evaluator(module.get(), _input_data_path, *metric.get());

  core::LayerParams layer_params;
  float int16_qerror =
    evaluate(evaluator, module_path, "int16" /* default quant_dtype */, layer_params);
  SolverOutput::get() << "Full int16 model qerror: " << int16_qerror << "\n";

  float uint8_qerror =
    evaluate(evaluator, module_path, "uint8" /* default quant_dtype */, layer_params);
  SolverOutput::get() << "Full uint8 model qerror: " << uint8_qerror << "\n";
  _quantizer->set_hook(_hooks.get());
  if (_hooks)
  {
    _hooks->on_begin_solver(module_path, uint8_qerror, int16_qerror);
  }

  if (int16_qerror > uint8_qerror)
  {
    throw std::runtime_error("Q8 model's qerror is less than Q16 model's qerror.");
  }

  _qerror = int16_qerror + _qerror_ratio * std::fabs(uint8_qerror - int16_qerror);
  SolverOutput::get() << "Target qerror: " << _qerror << "\n";

  // it'is assumed that int16_qerror <= _qerror <= uint8_qerror,
  if (int16_qerror >= _qerror)
  {
    // return Q16 model (we can not make it more accurate)
    if (!_quantizer->quantize(module.get(), "int16", layer_params))
    {
      std::cerr << "ERROR: Failed to quantize model" << std::endl;
      return nullptr;
    }

    if (_hooks)
    {
      _hooks->on_end_solver(layer_params, "int16", int16_qerror);
    }

    SolverOutput::get() << "The best configuration is int16 configuration\n";
    return module;
  }
  else if (uint8_qerror <= _qerror)
  {
    // return Q8 model (we can not make it less accurate)
    if (!_quantizer->quantize(module.get(), "uint8", layer_params))
    {
      std::cerr << "ERROR: Failed to quantize model" << std::endl;
      return nullptr;
    }

    if (_hooks)
    {
      _hooks->on_end_solver(layer_params, "uint8", uint8_qerror);
    }

    SolverOutput::get() << "The best configuration is uint8 configuration\n";
    return module;
  }

  // search for optimal mixed precision quantization configuration
  int last_depth = -1;
  float best_depth = -1;
  float best_error = -1; // minimal error
  core::LayerParams best_params;
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
    case Algorithm::Auto:
      int16_front =
        front_has_higher_error(nodes_depth, _visq_data_path, 0.5f * (max_depth + min_depth));
      break;
    case Algorithm::ForceQ16Front:
      SolverOutput::get() << "Front part will be Q16, while the rear will be Q8\n";
      int16_front = true;
      break;
    case Algorithm::ForceQ16Back:
      SolverOutput::get() << "Front part will be Q8, while the rear will be Q16\n";
      int16_front = false;
      break;
  }

  SolverOutput::get() << "\n";

  while (true)
  {
    if (_hooks)
    {
      _hooks->on_begin_iteration();
    }

    int cut_depth = static_cast<int>(std::floor(0.5f * (min_depth + max_depth)));

    if (last_depth == cut_depth)
    {
      break;
    }

    SolverOutput::get() << "Looking for the optimal configuration in [" << min_depth << " , "
                        << max_depth << "] depth segment\n";

    last_depth = cut_depth;

    core::LayerParams layer_params;
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
        auto layer_param = std::make_shared<core::LayerParam>();
        {
          layer_param->name = cur_node->name();
          layer_param->dtype = "int16";
          layer_param->granularity = "channel";
        }

        layer_params.emplace_back(layer_param);
      }
    }

    float cur_error = evaluate(evaluator, module_path, "uint8", layer_params);

    if (_hooks)
    {
      _hooks->on_end_iteration(layer_params, "uint8", cur_error);
    }

    if (cur_error < _qerror)
    {
      SolverOutput::get() << "Qerror at depth " << cut_depth << " is " << cur_error
                          << " < target qerror (" << _qerror << ")\n";
      int16_front ? (max_depth = cut_depth) : (min_depth = cut_depth);
      best_params = layer_params;
      best_depth = cut_depth;
      best_error = cur_error;
    }
    else
    {
      SolverOutput::get() << "Qerror at depth " << cut_depth << " is " << cur_error
                          << (cur_error > _qerror ? " > " : " == ") << "target qerror (" << _qerror
                          << ")\n";
      int16_front ? (min_depth = cut_depth) : (max_depth = cut_depth);
    }
  }

  if (_hooks)
  {
    _hooks->on_end_solver(best_params, "uint8", best_error);
  }

  SolverOutput::get() << "Found the best configuration at depth " << best_depth << "\n";
  if (!_quantizer->quantize(module.get(), "uint8", best_params))
  {
    std::cerr << "ERROR: Failed to quantize model" << std::endl;
    return nullptr;
  }

  return module;
}
