/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
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

#include "luci/Pass/QuantizeDequantizeWeightsWithGPTQPass.h"
#include "QuantizationUtils.h"
#include "helpers/LayerInfoMap.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>
#include <loco/IR/TensorShape.h>

#include <cmath>
#include <functional>

namespace luci
{

namespace
{

using IterFunc = std::function<void(uint32_t *, loco::TensorShape &, int32_t)>;

void iterate_per_channel(CircleConst *node, IterFunc func)
{
  loco::TensorShape dimension;
  dimension.rank(4);
  uint32_t indices[4] = {
    0,
  };
  int32_t channel_dim_index{0};

  if (!get_channel_dim_index(node, dimension, channel_dim_index))
  {
    assert(false);
    return;
  }

  for (indices[0] = 0; indices[0] < dimension.dim(0).value(); indices[0]++)
  {
    for (indices[1] = 0; indices[1] < dimension.dim(1).value(); indices[1]++)
    {
      for (indices[2] = 0; indices[2] < dimension.dim(2).value(); indices[2]++)
      {
        for (indices[3] = 0; indices[3] < dimension.dim(3).value(); indices[3]++)
        {
          func(indices, dimension, channel_dim_index);
        }
      }
    }
  }
}

size_t calculate_qauntized_value(CircleConst *node, uint32_t *indices, loco::TensorShape &dimension,
                                 int index_channel_dim, std::vector<float> &scaling_factor,
                                 std::vector<float> &max, std::vector<float> &min)
{
  assert(node != nullptr);

  int idx_channel = indices[index_channel_dim];

  assert(scaling_factor[idx_channel] > 0);

  const float scaling_factor_inv = 1.0 / scaling_factor[idx_channel];
  auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
  auto data_clipped = std::min(std::max(data, max[idx_channel]), min[idx_channel]);

  return static_cast<int32_t>(std::round((data_clipped - min[idx_channel]) * scaling_factor_inv));
}

void cal_minmax_per_channel(CircleConst *node, std::vector<float> &min, std::vector<float> &max)
{
  loco::TensorShape dimension;
  dimension.rank(4);
  int32_t index_channel_dim{0};

  if (!get_channel_dim_index(node, dimension, index_channel_dim))
  {
    throw std::runtime_error("GPTQPass: Failed to get channel dim index.");
  }
  auto size = dimension.dim(index_channel_dim).value();

  std::vector<bool> has_min_max_value(size, false);
  min.resize(size);
  max.resize(size);

  auto cal_minmax = [&](uint32_t *indices, loco::TensorShape &dimension, int index_channel_dim) {
    int idx_channel = indices[index_channel_dim];
    auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
    if (has_min_max_value[idx_channel])
    {
      min[idx_channel] = std::min(data, min[idx_channel]);
      max[idx_channel] = std::max(data, max[idx_channel]);
    }
    else
    {
      min[idx_channel] = data;
      max[idx_channel] = data;
      has_min_max_value[idx_channel] = true;
    }
  };

  iterate_per_channel(node, cal_minmax);
}

/**
 * @brief Compute the scale and zero point for the given range of values
 */
void compute_asym_scale_zp(float min, float max, loco::DataType data_type, float &scaling_factor,
                           int64_t &zp, float &nudged_min, float &nudged_max)
{
  LOGGER(l);

  assert(min <= max);

  const int32_t kMinScale = 0;
  const int32_t kMaxScale = data_type == loco::DataType::U4 ? 15 : 255;

  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const double rmin = std::fmin(0, min);
  const double rmax = std::fmax(0, max);
  const double qrange = qmax_double - qmin_double;
  assert(qrange > 0);

  double scale = (rmax - rmin) / qrange;
  double zero_point_double = 0;
  uint8_t nudged_zero_point = 0;

  if (scale == 0)
  {
    WARN(l) << "GPTQPass: The minimum and maximum values are the same." << std::endl;
    if (min >= 0 && max >= 0)
      zero_point_double = kMinScale;
    else
      zero_point_double = kMaxScale;
  }
  else
    zero_point_double = qmin_double - rmin / scale;
  if (min >= 0)
  {
    assert(min >= 0 && max >= 0);
    nudged_zero_point = kMinScale;
    scale = max / qrange;
    if (min > 0 && max > 0)
      WARN(l) << "GPTQPass: The minimum and maximum values are all positive." << std::endl;
  }
  else if (max < 0)
  {
    assert(min < 0 && max < 0);
    nudged_zero_point = kMaxScale;
    scale = -min / qrange;
    WARN(l) << "GPTQPass: The minimum and maximum values are all negative." << std::endl;
  }
  else
  {
    assert(min < 0 && max >= 0);
    nudged_zero_point = fp32_to_uint8_cast(std::round(zero_point_double));
  }

  // protect scale from being very low due to overflow
  if (scale < 1e-5)
  {
    scale = 1e-5;
    nudged_zero_point = fp32_to_uint8_cast(std::round(qmin_double - rmin / scale));
  }

  nudged_min = static_cast<float>((qmin_double - nudged_zero_point) * scale);
  nudged_max = static_cast<float>((qmax_double - nudged_zero_point) * scale);

  scaling_factor = scale;
  zp = nudged_zero_point;
}

void asymmetric_wquant_per_channel(CircleConst *node, std::vector<float> &min,
                                   std::vector<float> &max, std::vector<float> &scaling_factor,
                                   std::vector<int64_t> &zp, std::vector<float> &nudged_min,
                                   std::vector<float> &nudged_max, loco::DataType output_type)
{
  assert(node->dtype() == loco::DataType::FLOAT32);
  assert(output_type == loco::DataType::U8 || output_type == loco::DataType::U4);

  const int32_t kMinScale = 0;
  const int32_t kMaxScale = output_type == loco::DataType::U4 ? 15 : 255;

  uint32_t input_size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(input_size);

  for (size_t i = 0; i < min.size(); ++i)
  {
    compute_asym_scale_zp(min[i], max[i], output_type, scaling_factor[i], zp[i], nudged_min[i],
                          nudged_max[i]);
  }

  auto quantize = [&](uint32_t *indices, loco::TensorShape &dimension, int index_channel_dim) {
    quantized_values[cal_offset(dimension, indices)] = calculate_qauntized_value(
      node, indices, dimension, index_channel_dim, scaling_factor, nudged_max, nudged_min);
  };
  iterate_per_channel(node, quantize);

  node->dtype(loco::DataType::U8);            // Change the type of tensor
  node->size<loco::DataType::U8>(input_size); // Resize tensor
  for (uint32_t i = 0; i < input_size; ++i)
  {
    node->at<loco::DataType::U8>(i) = std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void asymmetric_wdequant_per_channel(CircleConst *node, std::vector<float> &scaling_factor,
                                     std::vector<float> &nudged_min)
{
  assert(node->dtype() == loco::DataType::U8);
  uint32_t size = node->size<loco::DataType::U8>();
  std::vector<float> dequantized_values(size);

  auto dequantize = [&](uint32_t *indices, loco::TensorShape &dimension, int index_channel_dim) {
    int idx_channel = indices[index_channel_dim];
    auto data = node->at<loco::DataType::U8>(cal_offset(dimension, indices));
    dequantized_values[cal_offset(dimension, indices)] =
      static_cast<float>(data) * scaling_factor[idx_channel] + nudged_min[idx_channel];
  };

  iterate_per_channel(node, dequantize);

  node->dtype(loco::DataType::FLOAT32);      // change the type of tensor
  node->size<loco::DataType::FLOAT32>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::FLOAT32>(i) = dequantized_values[i];
  }
}

/**
 * @brief QuantizeWeightsWithGPTQ quantizes and dequantizes tensors for weights uisng GPTQ algorithm
 * @details Compensate for the quantization error and update weights using Hessian matrix
 *
 */
class QuantizeDequantizeWeightsWithGPTQ final : public luci::CircleNodeMutableVisitor<void>
{
public:
  QuantizeDequantizeWeightsWithGPTQ(
    loco::DataType input, loco::DataType output, QuantizationGranularity granularity,
    std::unordered_map<const luci::CircleNode *, std::vector<float>> *hessian_map)
    : _input_type(input), _output_type(output), _granularity(granularity), _hessian_map(hessian_map)
  {
  }

private:
  loco::DataType _input_type;
  loco::DataType _output_type;
  QuantizationGranularity _granularity;
  std::unordered_map<const luci::CircleNode *, std::vector<float>> *_hessian_map;

  void fake_quantize(luci::CircleConst *weights)
  {
    if (_granularity != luci::QuantizationGranularity::ChannelWise)
    {
      throw std::invalid_argument("GPTQPass: Unsupported granularity");
    }

    if (_output_type != loco::DataType::U4 && _output_type != loco::DataType::U8)
    {
      throw std::runtime_error("GPTQPass: GPTQ quantization supports uint4/uint8");
    }

    // Find min/max per channel
    std::vector<float> min;
    std::vector<float> max;

    cal_minmax_per_channel(weights, min, max);

    std::vector<float> nudged_min(min.size());
    std::vector<float> nudged_max(min.size());
    std::vector<float> scaling_factor(min.size());
    std::vector<int64_t> zp(min.size());

    asymmetric_wquant_per_channel(weights, min, max, scaling_factor, zp, nudged_min, nudged_max,
                                  _output_type);
    asymmetric_wdequant_per_channel(weights, scaling_factor, nudged_min);

    auto quantparam = std::make_unique<CircleQuantParam>();
    quantparam->min = nudged_min;
    quantparam->max = nudged_max;
    quantparam->scale = scaling_factor;
    quantparam->zerop = zp;

    weights->quantparam(std::move(quantparam));
  }

  void fake_quantize_with_gptq(luci::CircleConst *weights, std::vector<float> &hessian)
  {
    // To be implemented
    (void)weights;
    (void)hessian;
  }

private:
  // Check if
  // 1. node is const
  // 2. node's dtype is float32
  bool is_quantizable(loco::Node *node)
  {
    auto const_node = dynamic_cast<luci::CircleConst *>(node);
    if (not const_node)
      return false;

    // Skip if this is not float32
    if (const_node->dtype() != loco::DataType::FLOAT32)
      return false;

    return true;
  }

  // Default behavior (Do nothing)
  void visit(luci::CircleNode *) {}

  void visit(luci::CircleConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    auto hessian = (*_hessian_map)[node];

    fake_quantize_with_gptq(new_weights, hessian);
  }

  void visit(luci::CircleDepthwiseConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    fake_quantize(new_weights);
  }

  void visit(luci::CircleTransposeConv *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    fake_quantize(new_weights);
  }

  void visit(luci::CircleFullyConnected *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;
    if (not is_quantizable(node->weights()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->weights());
    auto new_weights = luci::clone(weights);
    node->weights(new_weights);

    auto hessian = (*_hessian_map)[node];

    fake_quantize_with_gptq(new_weights, hessian);
  }
};

} // namespace

bool QuantizeDequantizeWeightsWithGPTQPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "QuantizeDequantizeWeightsWithGPTQ Start" << std::endl;

  if (_ctx->input_model_dtype != loco::DataType::FLOAT32)
    throw std::runtime_error("GPTQPass: Weights-only quantization supports float32 input only");

  if (_ctx->output_model_dtype != loco::DataType::U8 &&
      _ctx->output_model_dtype != loco::DataType::U4)
  {
    throw std::runtime_error("GPTQPass: GPTQ quantization supports uint4/uint8");
  }

  auto info_by_name = layer_info_map(g, _ctx->layers_info);

  auto quantize_dtype = [&](const luci::CircleNode *node) {
    auto iter = info_by_name.find(node->name());

    // Return designated quantization dtype
    if (iter != info_by_name.end())
      return iter->second.dtype;

    // Return default quantization dtype
    return _ctx->output_model_dtype;
  };

  auto quantize_granularity = [&](const luci::CircleNode *node) {
    auto iter = info_by_name.find(node->name());

    // Return designated quantization granularity
    if (iter != info_by_name.end())
      return iter->second.granularity;

    // Return default quantization granularity
    return _ctx->granularity;
  };

  // Quantize weights
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    QuantizeDequantizeWeightsWithGPTQ qw(_ctx->input_model_dtype, quantize_dtype(circle_node),
                                         quantize_granularity(circle_node), _hessian_map);
    circle_node->accept(&qw);
  }

  INFO(l) << "QuantizeDequantizeWeightsWithGPTQ End" << std::endl;
  return false; // one time run
}

} // namespace luci
