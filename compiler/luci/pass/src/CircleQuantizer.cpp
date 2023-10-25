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
#include "luci/Pass/PropagateQParamForwardPass.h"
#include "luci/Pass/RequantizePass.h"
#include "luci/Pass/ConvertToFakeQuantizedModelPass.h"
#include "luci/Pass/FoldDequantizePass.h"
#include "luci/Pass/RemoveRedundantDequantizePass.h"
#include "luci/Pass/QuantizePreCheckerPass.h"
#include "luci/Pass/QuantizeWithMinMaxPass.h"
#include "luci/Pass/QuantizeDequantizeWeightsPass.h"
#include "luci/Pass/QuantizeWeightsPass.h"

#include "luci/Pass/CircleShapeInferencePass.h"
#include "luci/Pass/CircleTypeInferencePass.h"

// logo passes
#include <logo/RemoveDeadNodeWithQueryPass.h>

#include "ProgressReporter.h"
#include "helpers/Strings.h"

#include "QuantizedModelVerifier.h"

#include <luci/IR/CircleNode.h>
#include <logo/Phase.h>
#include <pepper/csv2vec.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_set>

namespace
{

using namespace luci;
using LayerParam = luci::CircleQuantizer::Options::LayerParam;
using LayerParams = luci::CircleQuantizer::Options::LayerParams;
using LayerParamsSet = luci::CircleQuantizer::Options::LayerParamsSet;

// This function updates user-given input_type to match with the input signature of graph
// If user gives only one input_type, it will be expanded to the number of graph inputs
void canonicalize_input_type(loco::Graph *g, std::vector<loco::DataType> &input_type)
{
  if (g == nullptr)
    return;

  const auto inputs = g->inputs();

  assert(inputs); // FIX_CALLER_UNLESS

  // Check validity of the number of input dtype given by a user
  if (input_type.size() != 1 and input_type.size() != inputs->size())
  {
    throw std::runtime_error(
      "Invalid number of input dtype. The number of input dtype should be 1 or "
      "the same as the number of graph inputs.");
  }

  // Handle the case when a user gives only one input dtype
  if (input_type.size() == 1)
  {
    const auto user_given_dtype = input_type[0];
    input_type.clear();

    // Expand input dtype to the number of graph inputs
    // Since quantizer can only quantize float32, user_given_dtype is set only for float32 inputs
    auto input_nodes = loco::input_nodes(g);
    for (uint32_t i = 0; i < input_nodes.size(); i++)
    {
      auto input = loco::must_cast<luci::CircleInput *>(input_nodes[i]);

      if (input->dtype() == loco::DataType::FLOAT32)
        input_type.push_back(user_given_dtype);
      else
        input_type.push_back(input->dtype());
    }
  }

  // Finally, check validity of input_type
  // input_type is valid if
  // C1. for non-float32 model input, input_type == model's input dtype
  // or
  // C2. for float32 model input, input_type == uint8, int16, or float32
  auto input_nodes = loco::input_nodes(g);
  for (uint32_t i = 0; i < input_nodes.size(); i++)
  {
    auto input = loco::must_cast<luci::CircleInput *>(input_nodes[i]);
    assert(i == input->index()); // FIX_ME_UNLESS

    if (input->dtype() != loco::DataType::FLOAT32)
    {
      // C1
      if (input->dtype() != input_type[i])
        throw std::runtime_error(
          "Input dtype of " + input->name() +
          " is invalid. It has to be the same with the model's input dtype.");
    }
    else
    {
      // C2
      if (input_type[i] != loco::DataType::FLOAT32 and input_type[i] != loco::DataType::U8 and
          input_type[i] != loco::DataType::S16)
      {
        throw std::runtime_error("Input dtype of " + input->name() +
                                 " is invalid. For float32 input, the input dtype after "
                                 "quantization must be one of uint8, int16, or float32.");
      }
    }
  }
}

// This function updates user-given output_type to match with the output signature of graph
// If user gives only one output_type, it will be expanded to the number of graph outputs
// NOTE This function is almost same with canonicalize_input_type, but it is written as a
// separate function for more precise error messaging.
// TODO Find a way to reduce duplicate codes
void canonicalize_output_type(loco::Graph *g, std::vector<loco::DataType> &output_type)
{
  if (g == nullptr)
    return;

  const auto outputs = g->outputs();

  assert(outputs); // FIX_CALLER_UNLESS

  // Check validity of the number of output dtype given by a user
  if (output_type.size() != 1 and output_type.size() != outputs->size())
  {
    throw std::runtime_error(
      "Invalid number of output dtype. The number of output dtype should be 1 or "
      "the same as the number of graph outputs.");
  }

  // Handle the case when a user gives only one output dtype
  if (output_type.size() == 1)
  {
    const auto user_given_dtype = output_type[0];
    output_type.clear();

    // Expand output dtype to the number of graph outputs
    // If dtype of graph output is float32, it will be replaced with user_given_dtype
    // Otherwise, it will not change
    auto output_nodes = loco::output_nodes(g);
    for (uint32_t i = 0; i < output_nodes.size(); i++)
    {
      auto output = loco::must_cast<luci::CircleOutput *>(output_nodes[i]);

      if (output->dtype() == loco::DataType::FLOAT32)
        output_type.push_back(user_given_dtype);
      else
        output_type.push_back(output->dtype());
    }
  }

  // Finally, check validity of output_type
  // output_type is valid if
  // C1. for non-float32 model output, output_type == model's output dtype
  // or
  // C2. for float32 model output, output_type == uint8, int16, or float32
  auto output_nodes = loco::output_nodes(g);
  for (uint32_t i = 0; i < output_nodes.size(); i++)
  {
    auto output = loco::must_cast<luci::CircleOutput *>(output_nodes[i]);
    assert(i == output->index()); // FIX_ME_UNLESS

    if (output->dtype() != loco::DataType::FLOAT32)
    {
      // C1
      if (output->dtype() != output_type[i])
        throw std::runtime_error(
          "Output dtype of " + output->name() +
          " is invalid. It has to be the same with the model's output dtype.");
    }
    else
    {
      // C2
      if (output_type[i] != loco::DataType::FLOAT32 and output_type[i] != loco::DataType::U8 and
          output_type[i] != loco::DataType::S16)
      {
        throw std::runtime_error("Output dtype of " + output->name() +
                                 " is invalid. For float32 output, the output dtype after "
                                 "quantization must be one of uint8, int16, or float32.");
      }
    }
  }
}

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
  void layer_params(AlgorithmParameters, LayerParams &) final;
  LayerParams layer_params(AlgorithmParameters) const final;
  void layer_params_set(LayerParamsSet &) final;
  LayerParamsSet layer_params_set(void) const final;
  bool query(Algorithm) final;

private:
  std::vector<Algorithm> _algorithms;
  std::map<AlgorithmParameters, const std::string> _algorithm_params;
  std::map<AlgorithmParameters, std::vector<std::string>> _multiple_params;
  std::map<AlgorithmParameters, LayerParams> _layer_params;
  LayerParamsSet _layer_params_set;
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

void QuantizeOptionsImpl::layer_params(AlgorithmParameters param, LayerParams &vec)
{
  _layer_params[param] = vec;
}

LayerParams QuantizeOptionsImpl::layer_params(AlgorithmParameters param) const
{
  auto param_vec = _layer_params.find(param);
  if (param_vec != _layer_params.end())
  {
    return param_vec->second;
  }
  else
  {
    return LayerParams();
  }
}

void QuantizeOptionsImpl::layer_params_set(LayerParamsSet &vec) { _layer_params_set = vec; }

LayerParamsSet QuantizeOptionsImpl::layer_params_set(void) const { return _layer_params_set; }

bool QuantizeOptionsImpl::query(Algorithm algo)
{
  std::vector<Algorithm>::iterator it = std::find(_algorithms.begin(), _algorithms.end(), algo);
  if (it == _algorithms.end())
    return false;

  return true;
}

} // namespace

namespace
{

bool is_valid_params(loco::Graph *g, LayerParams &lps)
{
  // no same name in lps
  std::unordered_set<std::string> us;
  for (auto &lp : lps)
  {
    if (us.find(lp->name) != us.end())
      throw std::runtime_error("Duplicate name found in configuration: " + lp->name);
    us.emplace(lp->name);
  }

  // all name should be found in graph
  for (auto &lp : lps)
  {
    auto name = lp->name;
    bool found = false;
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      auto cnode = loco::must_cast<luci::CircleNode *>(node);
      if (cnode->opcode() == luci::CircleOpcode::CIRCLEOUTPUT)
        continue;

      if (cnode->name() == name)
      {
        found = true;
        break;
      }
    }
    if (not found)
      return false;
  }
  return true;
}

LayerParams find_valid_params(loco::Graph *g, LayerParamsSet &lpss)
{
  // valid condition: there should be only one LayerParams that is OK
  uint32_t valid_count = 0;
  LayerParams params;
  for (auto &lps : lpss)
  {
    if (is_valid_params(g, lps))
    {
      valid_count++;
      params = lps;
    }
  }
  if (valid_count != 1)
    throw std::runtime_error(
      "Configuration file has layer names (and alternates) that can be mapped in multiple or no "
      "ways. Please update configuration file to have only one valid name mapping.");

  return params;
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
    auto layer_params = _options->layer_params(Options::AlgorithmParameters::Quantize_layer_params);
    auto layer_params_set = _options->layer_params_set();

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

    if (layer_params_set.size() > 1u)
    {
      layer_params = find_valid_params(g, layer_params_set);
    }

    // Check dtype/granularity of layer params
    for (auto layer_param : layer_params)
    {
      auto name = layer_param->name;
      if (!in_array(to_lower_case(layer_param->dtype), fakeq_supported_output_model_dtype))
      {
        throw std::runtime_error("Unsupported dtype in " + name + ". List of supported dtype: " +
                                 to_string(fakeq_supported_output_model_dtype));
      }
      if (!in_array(to_lower_case(layer_param->granularity), fakeq_supported_granularity))
      {
        throw std::runtime_error(
          "Unsupported granularity in " + name +
          ". List of supported granularity: " + to_string(fakeq_supported_granularity));
      }
    }

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

      for (auto layer_param : layer_params)
      {
        LayerInfo info;
        {
          info.name = layer_param->name;
          info.dtype = str_to_dtype(layer_param->dtype);
          info.granularity = str_to_granularity(layer_param->granularity);
        }
        ctx->layers_info.emplace_back(info);
      }
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
    static const std::vector<std::string> qwmm_supported_input_type{"uint8", "int16",   "int32",
                                                                    "int64", "float32", "bool"};
    static const std::vector<std::string> qwmm_supported_output_type{"uint8", "int16",   "int32",
                                                                     "int64", "float32", "bool"};

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

    auto input_type_vec = pepper::csv_to_vector<std::string>(input_type);
    auto output_type_vec = pepper::csv_to_vector<std::string>(output_type);

    bool TF_style_maxpool =
      _options->param(Options::AlgorithmParameters::Quantize_TF_style_maxpool) == "True";

    bool save_min_max =
      _options->param(Options::AlgorithmParameters::Quantize_save_min_max) == "True";

    auto layer_params = _options->layer_params(Options::AlgorithmParameters::Quantize_layer_params);
    auto layer_params_set = _options->layer_params_set();

    if (!in_array(to_lower_case(input_model_dtype), qwmm_supported_input_model_dtype))
      throw std::runtime_error("Unsupported input type. List of supported input types: " +
                               to_string(qwmm_supported_input_model_dtype));

    if (!in_array(to_lower_case(output_model_dtype), qwmm_supported_output_model_dtype))
      throw std::runtime_error("Unsupported output type. List of supported output types: " +
                               to_string(qwmm_supported_output_model_dtype));

    if (!in_array(to_lower_case(granularity), qwmm_supported_granularity))
      throw std::runtime_error("Unsupported granularity. List of supported granularity: " +
                               to_string(qwmm_supported_granularity));

    for (const auto &dtype : input_type_vec)
    {
      if (!in_array(to_lower_case(dtype), qwmm_supported_input_type))
        throw std::runtime_error("Unsupported input type. List of supported input types: " +
                                 to_string(qwmm_supported_input_type));
    }

    for (const auto &dtype : output_type_vec)
    {
      if (!in_array(to_lower_case(dtype), qwmm_supported_output_type))
        throw std::runtime_error("Unsupported output type. List of supported output types: " +
                                 to_string(qwmm_supported_output_type));
    }

    if (str_to_granularity(granularity) == QuantizationGranularity::LayerWise &&
        str_to_dtype(output_model_dtype) != loco::DataType::U8)
      throw std::runtime_error("Layer-wise quantization only supports uint8 dtype.");

    if (layer_params_set.size() > 1u)
    {
      layer_params = find_valid_params(g, layer_params_set);
    }

    // Check dtype/granularity of layer params
    for (auto layer_param : layer_params)
    {
      auto name = layer_param->name;
      if (!in_array(to_lower_case(layer_param->dtype), qwmm_supported_output_model_dtype))
      {
        throw std::runtime_error("Unsupported dtype in " + name + ". List of supported dtype: " +
                                 to_string(qwmm_supported_output_model_dtype));
      }
      if (!in_array(to_lower_case(layer_param->granularity), qwmm_supported_granularity))
      {
        throw std::runtime_error(
          "Unsupported granularity in " + name +
          ". List of supported granularity: " + to_string(qwmm_supported_granularity));
      }
    }

    auto input_types = str_vec_to_dtype_vec(input_type_vec);
    auto output_types = str_vec_to_dtype_vec(output_type_vec);

    // Canonicalize user-given input/output_type (match with # of inputs/outputs)
    canonicalize_input_type(g, input_types);
    canonicalize_output_type(g, output_types);

    // Input model checker for quantization
    luci::QuantizePreCheckerPass input_model_checker{};
    input_model_checker.run(g);

    auto ctx = std::make_unique<luci::QuantizeWithMinMaxPass::Context>();
    {
      ctx->input_model_dtype = str_to_dtype(input_model_dtype);
      ctx->output_model_dtype = str_to_dtype(output_model_dtype);
      ctx->granularity = str_to_granularity(granularity);
      ctx->input_types = input_types;
      ctx->output_types = output_types;
      ctx->TF_style_maxpool = TF_style_maxpool;
      ctx->save_min_max = save_min_max;

      for (auto layer_param : layer_params)
      {
        LayerInfo info;
        {
          info.name = layer_param->name;
          info.dtype = str_to_dtype(layer_param->dtype);
          info.granularity = str_to_granularity(layer_param->granularity);
        }
        ctx->layers_info.emplace_back(info);
      }
    }

    luci::QuantizeWithMinMaxPass quantizer(std::move(ctx));

    quantizer.run(g);

    auto verify_ctx = std::make_unique<luci::QuantizedModelVerifier::Context>();
    {
      verify_ctx->output_model_dtype = str_to_dtype(output_model_dtype);
      verify_ctx->granularity = str_to_granularity(granularity);
      verify_ctx->input_types = input_types;
      verify_ctx->output_types = output_types;
      verify_ctx->TF_style_maxpool = TF_style_maxpool;

      for (auto layer_param : layer_params)
      {
        LayerInfo info;
        {
          info.name = layer_param->name;
          info.dtype = str_to_dtype(layer_param->dtype);
          info.granularity = str_to_granularity(layer_param->granularity);
        }
        verify_ctx->layers_info.emplace_back(info);
      }
    }

    // Verify the type/granularity of the quantized model
    luci::QuantizedModelVerifier verifier(std::move(verify_ctx));

    verifier.verify(g);
  }

  if (_options->query(Options::Algorithm::QuantizeWeights))
  {
    static const std::vector<std::string> qw_supported_input_model_dtype{"float32"};
    static const std::vector<std::string> qw_supported_output_model_dtype{"int8", "int16"};
    static const std::vector<std::string> qw_supported_granularity{"channel"};

    auto input_model_dtype =
      _options->param(Options::AlgorithmParameters::Quantize_input_model_dtype);
    auto output_model_dtype =
      _options->param(Options::AlgorithmParameters::Quantize_output_model_dtype);
    auto granularity = _options->param(Options::AlgorithmParameters::Quantize_granularity);

    if (!in_array(to_lower_case(input_model_dtype), qw_supported_input_model_dtype))
      throw std::runtime_error("Unsupported input type. List of supported input type: " +
                               to_string(qw_supported_input_model_dtype));

    if (!in_array(to_lower_case(output_model_dtype), qw_supported_output_model_dtype))
      throw std::runtime_error("Unsupported output type. List of supported output type: " +
                               to_string(qw_supported_output_model_dtype));

    if (!in_array(to_lower_case(granularity), qw_supported_granularity))
      throw std::runtime_error("Unsupported granularity. List of supported granularity: " +
                               to_string(qw_supported_granularity));
    auto ctx = std::make_unique<luci::QuantizeWeightsPass::Context>();
    {
      ctx->input_model_dtype = str_to_dtype(input_model_dtype);
      ctx->output_model_dtype = str_to_dtype(output_model_dtype);
      ctx->granularity = str_to_granularity(granularity);
    }
    luci::QuantizeWeightsPass weights_quantizer(std::move(ctx));

    weights_quantizer.run(g);
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

  // Convert quantized model to fake-quantized model
  if (_options->query(Options::Algorithm::ConvertToFakeQuantizedModel))
  {
    luci::ConvertToFakeQuantizedModelPass fake_quantizer;
    fake_quantizer.run(g);

    logo::Phase phase;

    // Default passes
    phase.emplace_back(std::make_unique<logo::RemoveDeadNodeWithQueryPass>());
    phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
    phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

    // Remove redundant Dequantize Ops generated during fake quantization
    phase.emplace_back(std::make_unique<luci::RemoveRedundantDequantizePass>());
    // Fold Dequantize Ops generated during fake quantization
    phase.emplace_back(std::make_unique<luci::FoldDequantizePass>());

    ProgressReporter prog(g, logo::PhaseStrategy::Restart);
    logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
    phase_runner.attach(&prog);
    phase_runner.run(phase);
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
