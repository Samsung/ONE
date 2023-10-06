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

#include "MPQSolver.h"

#include "pattern/PatternResolver.h"

#include <luci/ImporterEx.h>
#include <iostream>

using namespace mpqsolver;

using LayerParam = luci::CircleQuantizer::Options::LayerParam;
using LayerParams = luci::CircleQuantizer::Options::LayerParams;

MPQSolver::MPQSolver(const std::string &input_data_path, float qerror_ratio,
                     const std::string &input_quantization, const std::string &output_quantization)
  : _input_data_path(input_data_path), _qerror_ratio(qerror_ratio),
    _input_quantization(input_quantization), _output_quantization(output_quantization)
{
  _quantizer = std::make_unique<core::Quantizer>(_input_quantization, _output_quantization);
}

void MPQSolver::set_save_intermediate(const std::string &save_path)
{
  _hooks = std::make_unique<core::DumpingHooks>(save_path);
}

void MPQSolver::set_mpq_options(MPQOptions &options) { _options = options; }

LayerParams MPQSolver::get_frozen_params() const
{
  LayerParams params;
  for (auto node_to_param : _frozen._node_to_param)
  {
    params.push_back(std::make_shared<LayerParam>(node_to_param.second));
  }

  return params;
}

void MPQSolver::resolve_patterns(luci::Module *module)
{
  _frozen._node_to_param.clear();

  for (auto pattern : _options._patterns)
  {
    std::unique_ptr<pattern::PatternResolver> resolver;
    switch (pattern)
    {
      case QuantizationPattern::Q8LayerNormWithQ16Variance:
        resolver = std::make_unique<pattern::Q8LayerNormWithQ16VarianceResolver>();
        break;
      default:
        throw std::runtime_error("Unsupported pattern to resolve");
    }

    auto const resolved = resolver->resolve(module);
    for (auto node_param : resolved)
    {
      auto const frozen = _frozen._node_to_param.find(node_param.first);
      if (frozen == _frozen._node_to_param.end())
      {
        // node was not previously defined - just set it (no ambiguity)
        _frozen._node_to_param[node_param.first] = node_param.second;
      }
      else if (frozen->second.dtype != node_param.second.dtype)
      {
        // ambiguity (incoming description conflicts with current)
        throw std::runtime_error("Resolved patterns contradict each other");
      }
    }
  }
}

std::unique_ptr<luci::Module> MPQSolver::read_module(const std::string &path)
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
