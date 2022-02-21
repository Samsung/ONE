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

#include "luci/Pass/CopyQuantParamPass.h"
#include "luci/Pass/ForceQuantParamPass.h"
#include "luci/Pass/PropagateQuantParamPass.h"
#include "luci/Pass/RequantizePass.h"
#include "luci/Pass/QuantizeWithMinMaxPass.h"
#include "luci/Pass/QuantizeDequantizeWeightsPass.h"

#include "luci/Pass/CircleShapeInferencePass.h"
#include "luci/Pass/CircleTypeInferencePass.h"

// logo passes
#include <logo/RemoveDeadNodeWithQueryPass.h>

#include "ProgressReporter.h"
#include "helpers/Strings.h"

#include "QuantizedModelVerifier.h"

#include <luci/IR/CircleNode.h>
#include <logo/Phase.h>

#include <memory>

namespace
{

using namespace luci;

template <typename T> T lexical_cast(const std::string &str)
{
  std::istringstream ss;
  ss.str(str);
  T data;
  ss >> data;
  return data;
}

template <typename T> std::vector<T> lexical_cast(std::vector<std::string> &sv)
{
  std::vector<T> result;
  std::transform(sv.begin(), sv.end(), std::back_inserter(result),
                 [](std::string str) -> T { return lexical_cast<T>(str); });
  return result;
}

class QuantizeOptionsImpl final : public luci::CircleQuantizer::Options
{
public:
  void enable(Algorithm) final;
  void param(AlgorithmParameters, const std::string &) final;
  const std::string param(AlgorithmParameters) const final;
  void params(AlgorithmParameters, std::vector<std::string> &) final;
  std::vector<std::string> params(AlgorithmParameters) const final;
  bool query(Algorithm) final;

private:
  std::vector<Algorithm> _algorithms;
  std::map<AlgorithmParameters, const std::string> _algorithm_params;
  std::map<AlgorithmParameters, std::vector<std::string>> _multiple_params;
};

void QuantizeOptionsImpl::enable(Algorithm algo) { _algorithms.push_back(algo); }

void QuantizeOptionsImpl::param(AlgorithmParameters param, const std::string &str)
{
  _algorithm_params.insert(std::pair<AlgorithmParameters, const std::string>(param, str));
}

const std::string QuantizeOptionsImpl::param(AlgorithmParameters param) const
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

void QuantizeOptionsImpl::params(AlgorithmParameters param, std::vector<std::string> &vec)
{
  _multiple_params[param] = vec;
}

std::vector<std::string> QuantizeOptionsImpl::params(AlgorithmParameters param) const
{
  auto param_vec = _multiple_params.find(param);
  if (param_vec != _multiple_params.end())
  {
    return param_vec->second;
  }
  else
  {
    return std::vector<std::string>();
  }
}

bool QuantizeOptionsImpl::query(Algorithm algo)
{
  std::vector<Algorithm>::iterator it = std::find(_algorithms.begin(), _algorithms.end(), algo);
  if (it == _algorithms.end())
    return false;

  return true;
}

} // namespace

namespace luci
{

CircleQuantizer::Options *CircleQuantizer::options(void)
{
  if (_options == nullptr)
  {
    _options = std::make_unique<QuantizeOptionsImpl>();
  }

  return _options.get();
}

void CircleQuantizer::quantize(loco::Graph *g) const
{
  // Fake quantization of weights
  if (_options->query(Options::Algorithm::QuantizeDequantizeWeights))
  {
    static const std::vector<std::string> fakeq_supported_input_model_dtype{"float32"};
    static const std::vector<std::string> fakeq_supported_output_model_dtype{"uint8", "int16"};
    static const std::vector<std::string> fakeq_supported_granularity{"layer", "channel"};

    auto input_model_dtype =
      _options->param(Options::AlgorithmParameters::Quantize_input_model_dtype);
    auto output_model_dtype =
      _options->param(Options::AlgorithmParameters::Quantize_output_model_dtype);
    auto granularity = _options->param(Options::AlgorithmParameters::Quantize_granularity);

    if (!in_array(to_lower_case(input_model_dtype), fakeq_supported_input_model_dtype))
      throw std::runtime_error("Unsupported input type. List of supported input type: " +
                               to_string(fakeq_supported_input_model_dtype));

    if (!in_array(to_lower_case(output_model_dtype), fakeq_supported_output_model_dtype))
      throw std::runtime_error("Unsupported output type. List of supported output type: " +
                               to_string(fakeq_supported_output_model_dtype));

    if (!in_array(to_lower_case(granularity), fakeq_supported_granularity))
      throw std::runtime_error("Unsupported granularity. List of supported granularity: " +
                               to_string(fakeq_supported_granularity));

    if (str_to_granularity(granularity) == QuantizationGranularity::LayerWise &&
        str_to_dtype(output_model_dtype) != loco::DataType::U8)
      throw std::runtime_error("Layer-wise quantization only supports uint8 dtype.");

    // Clear existing quantparams before doing fake quantization
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);
      if (circle_node->quantparam() != nullptr)
        circle_node->quantparam(nullptr);
    }

    auto ctx = std::make_unique<luci::QuantizeDequantizeWeightsPass::Context>();
    {
      ctx->input_model_dtype = str_to_dtype(input_model_dtype);
      ctx->output_model_dtype = str_to_dtype(output_model_dtype);
      ctx->granularity = str_to_granularity(granularity);
    }

    luci::QuantizeDequantizeWeightsPass fake_quantizer(std::move(ctx));

    fake_quantizer.run(g);
  }

  // Actual quantization of weights, bias, and activation
  if (_options->query(Options::Algorithm::QuantizeWithMinMax))
  {
    static const std::vector<std::string> qwmm_supported_input_model_dtype{"float32"};
    static const std::vector<std::string> qwmm_supported_output_model_dtype{"uint8", "int16"};
    static const std::vector<std::string> qwmm_supported_granularity{"layer", "channel"};
    static const std::vector<std::string> qwmm_supported_input_type{"uint8", "int16"};
    static const std::vector<std::string> qwmm_supported_output_type{"uint8", "int16"};

    auto input_model_dtype =
      _options->param(Options::AlgorithmParameters::Quantize_input_model_dtype);
    auto output_model_dtype =
      _options->param(Options::AlgorithmParameters::Quantize_output_model_dtype);
    auto granularity = _options->param(Options::AlgorithmParameters::Quantize_granularity);
    auto input_type = _options->param(Options::AlgorithmParameters::Quantize_input_type);
    if (input_type.empty())
      input_type = output_model_dtype;
    auto output_type = _options->param(Options::AlgorithmParameters::Quantize_output_type);
    if (output_type.empty())
      output_type = output_model_dtype;

    bool TF_style_maxpool =
      _options->param(Options::AlgorithmParameters::Quantize_TF_style_maxpool) == "True";

    if (!in_array(to_lower_case(input_model_dtype), qwmm_supported_input_model_dtype))
      throw std::runtime_error("Unsupported input type. List of supported input types: " +
                               to_string(qwmm_supported_input_model_dtype));

    if (!in_array(to_lower_case(output_model_dtype), qwmm_supported_output_model_dtype))
      throw std::runtime_error("Unsupported output type. List of supported output types: " +
                               to_string(qwmm_supported_output_model_dtype));

    if (!in_array(to_lower_case(granularity), qwmm_supported_granularity))
      throw std::runtime_error("Unsupported granularity. List of supported granularity: " +
                               to_string(qwmm_supported_granularity));

    if (!in_array(to_lower_case(input_type), qwmm_supported_input_type))
      throw std::runtime_error("Unsupported input type. List of supported input types: " +
                               to_string(qwmm_supported_input_type));

    if (!in_array(to_lower_case(output_type), qwmm_supported_output_type))
      throw std::runtime_error("Unsupported output type. List of supported output types: " +
                               to_string(qwmm_supported_output_type));

    if (str_to_granularity(granularity) == QuantizationGranularity::LayerWise &&
        str_to_dtype(output_model_dtype) != loco::DataType::U8)
      throw std::runtime_error("Layer-wise quantization only supports uint8 dtype.");

    auto ctx = std::make_unique<luci::QuantizeWithMinMaxPass::Context>();
    {
      ctx->input_model_dtype = str_to_dtype(input_model_dtype);
      ctx->output_model_dtype = str_to_dtype(output_model_dtype);
      ctx->granularity = str_to_granularity(granularity);
      ctx->input_type = str_to_dtype(input_type);
      ctx->output_type = str_to_dtype(output_type);
      ctx->TF_style_maxpool = TF_style_maxpool;
    }

    luci::QuantizeWithMinMaxPass quantizer(std::move(ctx));

    quantizer.run(g);

    // Verify the type/granularity of the quantized model
    luci::QuantizedModelVerifier verifier(str_to_dtype(output_model_dtype),
                                          str_to_granularity(granularity));
    verifier.verify(g);
  }

  // Requantize
  if (_options->query(Options::Algorithm::Requantize))
  {
    static const std::vector<std::string> rq_supported_input_model_dtype{"int8"};
    static const std::vector<std::string> rq_supported_output_model_dtype{"uint8"};

    auto input_model_dtype =
      _options->param(Options::AlgorithmParameters::Quantize_input_model_dtype);
    auto output_model_dtype =
      _options->param(Options::AlgorithmParameters::Quantize_output_model_dtype);

    if (!in_array(to_lower_case(input_model_dtype), rq_supported_input_model_dtype))
      throw std::runtime_error("Unsupported input type. List of supported input types: " +
                               to_string(rq_supported_input_model_dtype));

    if (!in_array(to_lower_case(output_model_dtype), rq_supported_output_model_dtype))
      throw std::runtime_error("Unsupported output type. List of supported output types: " +
                               to_string(rq_supported_output_model_dtype));

    luci::RequantizePass requantizer(str_to_dtype(input_model_dtype),
                                     str_to_dtype(output_model_dtype));
    requantizer.run(g);
  }

  // Force to write quantparam to specified tensors
  // NOTE Only per-tensor (not per-channel) qparam can be written
  if (_options->query(Options::Algorithm::ForceQuantParam))
  {
    ForceQuantParamPass::TensorVector tensors =
      _options->params(Options::AlgorithmParameters::Quantize_tensor_names);
    auto str_scales = _options->params(Options::AlgorithmParameters::Quantize_scales);
    auto str_zero_points = _options->params(Options::AlgorithmParameters::Quantize_zero_points);

    // Cast scales/zero_points to proper types
    ForceQuantParamPass::ScaleVector scales = lexical_cast<float>(str_scales);
    ForceQuantParamPass::ZPVector zero_points = lexical_cast<int64_t>(str_zero_points);

    ForceQuantParamPass fq(tensors, scales, zero_points);
    fq.run(g);
  }

  // Copy quantparam of a tensor to another tensor
  if (_options->query(Options::Algorithm::CopyQuantParam))
  {
    CopyQuantParamPass::TensorVector src_tensors =
      _options->params(Options::AlgorithmParameters::Quantize_src_tensor_names);
    CopyQuantParamPass::TensorVector dst_tensors =
      _options->params(Options::AlgorithmParameters::Quantize_dst_tensor_names);

    CopyQuantParamPass cq(src_tensors, dst_tensors);
    cq.run(g);
  }

  logo::Phase phase;

  // Do Shape/Type inference
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

  ProgressReporter prog(g, logo::PhaseStrategy::Saturate);
  logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

} // namespace luci
