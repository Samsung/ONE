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

#include "PatternSolver.h"

#include "PatternResolver.h"

#include <iostream>

using namespace mpqsolver::pattern;

using LayerParam = luci::CircleQuantizer::Options::LayerParam;
using LayerParams = luci::CircleQuantizer::Options::LayerParams;

PatternSolver::PatternSolver(const std::string &input_quantization,
                             const std::string &output_quantization,
                             const std::vector<QuantizationPattern> &patterns)
  : MPQSolver("", 1.f, input_quantization, output_quantization)
{
  MPQOptions options{patterns};
  set_mpq_options(options);
}

std::unique_ptr<luci::Module> PatternSolver::run(const std::string &module_path)
{
  auto module = read_module(module_path);
  assert(module != nullptr);

  resolve_patterns(module.get());

  auto layer_params = get_frozen_params();

  if (!_quantizer->quantize(module.get(), "uint8", layer_params))
  {
    throw std::runtime_error("Failed to quantize model");
  }

  return module;
}

void PatternSolver::set_mpq_options(MPQOptions &options) { _options = options; }

LayerParams PatternSolver::get_frozen_params() const
{
  LayerParams params;
  for (auto node_to_param : _frozen._node_to_param)
  {
    params.push_back(std::make_shared<LayerParam>(node_to_param.second));
  }

  return params;
}

void PatternSolver::resolve_patterns(luci::Module *module)
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
      case QuantizationPattern::Q8SoftmaxWithQ16SubExp:
        resolver = std::make_unique<pattern::Q8SoftmaxWithQ16SubExpResolver>();
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
