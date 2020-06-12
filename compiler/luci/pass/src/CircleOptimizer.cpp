/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/CircleOptimizer.h"

#include "luci/Pass/FuseBCQPass.h"
#include "luci/Pass/FuseInstanceNormPass.h"
#include "luci/Pass/FuseInstanceNormPassV2.h"
#include "luci/Pass/ResolveCustomOpAddPass.h"
#include "luci/Pass/ResolveCustomOpBatchMatMulPass.h"
#include "luci/Pass/QuantizeWithMinMaxPass.h"
#include "luci/Pass/QuantizeDequantizeWeightsPass.h"
// TODO add more passes

#include "luci/Pass/ShapeInferencePass.h"
#include "luci/Pass/TypeInferencePass.h"

// logo passes
#include <logo/RemoveDeadNodeWithQueryPass.h>

#include "ProgressReporter.h"
#include "CircleOptimizerUtils.h"

#include <logo/Phase.h>

#include <memory>

namespace
{

using namespace luci;

class OptimizeOptionsImpl final : public luci::CircleOptimizer::Options
{
public:
  void enable(Algorithm) final;
  void param(AlgorithmParameters, const std::string &) final;
  const std::string param(AlgorithmParameters) const final;
  bool query(Algorithm) final;

private:
  std::vector<Algorithm> _algorithms;
  std::map<AlgorithmParameters, const std::string> _algorithm_params;
};

void OptimizeOptionsImpl::enable(Algorithm algo) { _algorithms.push_back(algo); }

void OptimizeOptionsImpl::param(AlgorithmParameters param, const std::string &str)
{
  _algorithm_params.insert(std::pair<AlgorithmParameters, const std::string>(param, str));
}

const std::string OptimizeOptionsImpl::param(AlgorithmParameters param) const
{
  auto param_str = _algorithm_params.find(param);
  if (param_str != _algorithm_params.end())
  {
    return param_str->second;
  }
  else
  {
    return std::string();
  }
}

bool OptimizeOptionsImpl::query(Algorithm algo)
{
  std::vector<Algorithm>::iterator it = std::find(_algorithms.begin(), _algorithms.end(), algo);
  if (it == _algorithms.end())
    return false;

  return true;
}

} // namespace

namespace luci
{

CircleOptimizer::Options *CircleOptimizer::options(void)
{
  if (_options == nullptr)
  {
    _options = std::make_unique<OptimizeOptionsImpl>();
  }

  return _options.get();
}

void CircleOptimizer::optimize(loco::Graph *g) const
{
  logo::Phase phase;

  /* TRANSFORM DECLARATION BEGIN */
  if (_options->query(Options::Algorithm::ResolveCustomOpAdd))
  {
    phase.emplace_back(std::make_unique<luci::ResolveCustomOpAddPass>());
  }
  if (_options->query(Options::Algorithm::ResolveCustomOpBatchMatMul))
  {
    phase.emplace_back(std::make_unique<luci::ResolveCustomOpBatchMatMulPass>());
  }
  if (_options->query(Options::Algorithm::FuseInstanceNorm))
  {
    phase.emplace_back(std::make_unique<FuseInstanceNormPass>());
  }
  if (_options->query(Options::Algorithm::FuseInstanceNormV2))
  {
    phase.emplace_back(std::make_unique<FuseInstanceNormPassV2>());
  }
  if (_options->query(Options::Algorithm::FuseBCQ))
  {
    phase.emplace_back(std::make_unique<FuseBCQPass>());
  }

  // Shape inference is needed for added nodes doing above transformations
  phase.emplace_back(std::make_unique<luci::ShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::TypeInferencePass>());
  phase.emplace_back(std::make_unique<logo::RemoveDeadNodeWithQueryPass>());
  /* TRANSFORM DECLARATION END */

  ProgressReporter prog(g, logo::PhaseStrategy::Saturate);
  logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

void CircleOptimizer::quantize(loco::Graph *g) const
{
  // Fake quantization of weights
  if (_options->query(Options::Algorithm::QuantizeDequantizeWeights))
  {
    auto input_dtype = _options->param(Options::AlgorithmParameters::Quantize_input_dtype);
    auto output_dtype = _options->param(Options::AlgorithmParameters::Quantize_output_dtype);
    auto granularity = _options->param(Options::AlgorithmParameters::Quantize_granularity);

    luci::QuantizeDequantizeWeightsPass fake_quantizer(
        str_to_dtype(input_dtype), str_to_dtype(output_dtype), str_to_granularity(granularity));
    fake_quantizer.run(g);
  }

  // Actual quantization of weights, bias, and activation
  if (_options->query(Options::Algorithm::QuantizeWithMinMax))
  {
    auto input_dtype = _options->param(Options::AlgorithmParameters::Quantize_input_dtype);
    auto output_dtype = _options->param(Options::AlgorithmParameters::Quantize_output_dtype);
    auto granularity = _options->param(Options::AlgorithmParameters::Quantize_granularity);

    luci::QuantizeWithMinMaxPass quantizer(str_to_dtype(input_dtype), str_to_dtype(output_dtype),
                                           str_to_granularity(granularity));
    quantizer.run(g);
  }

  logo::Phase phase;

  // Do Shape/Type inference
  phase.emplace_back(std::make_unique<luci::ShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::TypeInferencePass>());

  ProgressReporter prog(g, logo::PhaseStrategy::Saturate);
  logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

} // namespace luci
