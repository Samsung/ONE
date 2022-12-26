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
#include "ErrorApproximator.h"
#include "Evaluator.h"
#include "Quantizer.h"
#include "DepthParameterizer.h"

#include <luci/ImporterEx.h>
#include <luci/Importer.h>
#include <luci/Log.h>

#include <cmath>
#include <iostream>

using namespace mpqsolver;

namespace
{

bool error_at_input_is_larger_than_at_output(const NodeDepthType &nodes_depth, float cut_depth)
{
  LOGGER(l);

  float error_at_input = 0;
  float error_at_output = 0;
  for (auto &iter : nodes_depth)
  {
    float cur_error = ErrorApproximator::approximate(iter.first);
    if (iter.second < cut_depth)
    {
      error_at_input += cur_error;
    }
    else
    {
      error_at_output += cur_error;
    }
  }

  if (error_at_input > error_at_output)
  {
    VERBOSE(l, 0) << "Q16 will be set at input due to ";
  }
  else
  {
    VERBOSE(l, 0) << "Q8 will be set at input due to ";
  }
  VERBOSE(l, 0) << error_at_input << " error at input vs ";
  VERBOSE(l, 0) << error_at_output << " error at output." << std::endl;

  return error_at_input > error_at_output;
}

class BisectionOptionsImpl final : public BisectionSolver::Options
{
public:
  virtual void enable(Q16AtInput) final;
  virtual bool query(Q16AtInput) final;

private:
  Q16AtInput _q16AtInput = Q16AtInput::Auto;
};

void BisectionOptionsImpl::enable(Q16AtInput q16AtInput) { _q16AtInput = q16AtInput; }
bool BisectionOptionsImpl::query(Q16AtInput q16AtInput) { return _q16AtInput == q16AtInput; }

std::unique_ptr<luci::Module> read_module(const std::string &path)
{
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(path);
  if (module.get() == nullptr)
  {
    std::cerr << "Failed to load " << path << std::endl;
    return nullptr;
  }

  return module;
}

float evaluate(const DatasetEvaluator &evaluator, const std::string &flt_path,
               const std::string &def_quant, LayerParams layers)
{
  auto model = read_module(flt_path);
  // get fake quantized model for evaluation
  if (!Quantizer::fake_quantize(model.get(), def_quant, layers))
  {
    throw std::runtime_error("Failed to produce fake-quantized model.");
  }

  return evaluator.evaluate(model.get());
}

} // namespace

BisectionSolver::BisectionSolver(const std::string &input_data_path, float qerror_ratio)
  : _input_data_path(input_data_path), _qerror_ratio(qerror_ratio), _qerror(0.f)
{
}

BisectionSolver::Options *BisectionSolver::options(void)
{
  if (_options == nullptr)
  {
    _options = std::make_unique<BisectionOptionsImpl>();
  }

  return _options.get();
}

std::unique_ptr<luci::Module> BisectionSolver::run(const std::string &flt_model_path)
{
  LOGGER(l);

  auto module = read_module(flt_model_path);

  float min_depth = 0.f;
  float max_depth = 0.f;
  NodeDepthType nodes_depth;
  if (DepthParameterizer::compute_depth(module.get(), nodes_depth, min_depth, max_depth) !=
      EXIT_SUCCESS)
  {
    std::cerr << "Invalid graph for bisectioning" << std::endl;
    return nullptr;
  }

  DatasetEvaluator evaluator(module.get(), _input_data_path);
  LayerParams layer_params;
  float int16_qerror = evaluate(evaluator, flt_model_path, "int16", layer_params);
  VERBOSE(l, 0) << "int16_quantization_error " << int16_qerror << std::endl;

  float int8_qerror = evaluate(evaluator, flt_model_path, "uint8", layer_params);
  VERBOSE(l, 0) << "int8_quantization_error " << int8_qerror << std::endl;

  _qerror = int16_qerror + _qerror_ratio * std::fabs(int8_qerror - int16_qerror);
  VERBOSE(l, 0) << "target quantization error " << _qerror << std::endl;

  if (int16_qerror > _qerror)
  {
    std::cerr << "ERROR: minimal targeted is an error of Q16 quantized model" << std::endl;
    return nullptr;
  }
  if (int8_qerror < _qerror)
  {
    // no need for bisectioning just return Q8 model
    if (!Quantizer::quantize(module.get(), "uint8", layer_params))
    {
      return nullptr;
    }

    return module;
  }

  int last_depth = -1;
  float best_depth = -1;
  LayerParams best_params;
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
  bool int16_at_input = true;
  if (auto option = options())
  {
    if (option->query(Options::Q16AtInput::Auto))
    {
      int16_at_input =
        error_at_input_is_larger_than_at_output(nodes_depth, 0.5f * (max_depth + min_depth));
    }
    else if (option->query(Options::Q16AtInput::True))
    {
      int16_at_input = true;
    }
    else if (option->query(Options::Q16AtInput::False))
    {
      int16_at_input = false;
    }
  }

  while (true)
  {
    float cut_depth = 0.5f * (min_depth + max_depth);

    if (last_depth == int(cut_depth))
    {
      break;
    }
    last_depth = int(cut_depth);

    auto nodes = graph->nodes();
    LayerParams layer_params;
    for (auto &node : active_nodes)
    {
      auto cur_node = loco::must_cast<luci::CircleNode *>(node);
      auto iter = nodes_depth.find(cur_node);
      if (iter == nodes_depth.end())
      {
        continue;
      }

      float depth = iter->second;
      if ((depth <= cut_depth && int16_at_input) || (depth >= cut_depth && !int16_at_input))
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

    float cur_accuracy = evaluate(evaluator, flt_model_path, "uint8", layer_params);
    VERBOSE(l, 0) << cut_depth << " : " << cur_accuracy << std::endl;

    if (cur_accuracy < _qerror)
    {
      int16_at_input ? (max_depth = cut_depth) : (min_depth = cut_depth);
      best_params = layer_params;
      best_depth = cut_depth;
    }
    else
    {
      int16_at_input ? (min_depth = cut_depth) : (max_depth = cut_depth);
    }
  }

  if (best_params.empty())
  {
    std::cerr << "Failed to find any configuration" << std::endl;
    return nullptr;
  }

  VERBOSE(l, 0) << "Found the best configuration at " << best_depth << " depth." << std::endl;
  if (!Quantizer::quantize(module.get(), "uint8", best_params))
  {
    return nullptr;
  }

  return module;
}
