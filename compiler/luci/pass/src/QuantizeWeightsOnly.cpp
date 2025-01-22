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

#include "QuantizeWeightsOnly.h"
#include "QuantizationUtils.h"

#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>

#include <cmath>
#include <vector>
#include <functional>
#include <limits>

using namespace luci;

namespace
{

using IterFunc = std::function<void(uint32_t *, loco::TensorShape &, int32_t)>;

void iterate_per_channel(CircleConst *node, int32_t &channel_dim_index, IterFunc func)
{
  loco::TensorShape dimension;
  dimension.rank(4);
  uint32_t indices[4] = {
    0,
  };

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

template <loco::DataType out_type>
void sym_wquant_per_channel_minimum_mse(CircleConst *node, std::vector<float> &min,
                                        std::vector<float> &max, std::vector<float> &scaling_factor,
                                        std::vector<float> &nudged_min,
                                        std::vector<float> &nudged_max, int32_t &channel_dim_index,
                                        const QuantizationAlgorithmParams &params)
{
  assert(node->dtype() == loco::DataType::FLOAT32);
  assert(out_type == loco::DataType::S4 || out_type == loco::DataType::S8 ||
         out_type == loco::DataType::S16);

  const auto kPhi = 1.618033988749894848204586834365638118; // Golden ratio
  const auto kSearchIterations = params.iterations_num;
  const auto kRangeCoefficient = params.range;

  const int32_t kMaxScale = max_for_sym_quant(out_type);
  const int32_t kMinScale = -kMaxScale;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(size);

  for (size_t i = 0; i < min.size(); ++i)
  {
    compute_sym_scale(min[i], max[i], scaling_factor[i], nudged_min[i], nudged_max[i], out_type);
  }

  auto quantize = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    const float scaling_factor_inv = 1.0 / scaling_factor[channel_idx];
    auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
    data = data < nudged_min[channel_idx] ? nudged_min[channel_idx] : data;
    data = data > nudged_max[channel_idx] ? nudged_max[channel_idx] : data;
    quantized_values[cal_offset(dimension, indices)] =
      static_cast<int32_t>(std::round(data * scaling_factor_inv));
  };
  std::vector<float> max_scale(min.size());
  for (size_t i = 0; i < min.size(); ++i)
  {
    max_scale[i] = std::max(std::fabs(min[i]), std::fabs(max[i]));
  }
  std::vector<double> channel_mse(min.size());
  std::vector<double> channel_min_mse(min.size(), std::numeric_limits<double>::max());

  auto calculate_mse = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
    data = data < nudged_min[channel_idx] ? nudged_min[channel_idx] : data;
    data = data > nudged_max[channel_idx] ? nudged_max[channel_idx] : data;
    double diff =
      data - quantized_values[cal_offset(dimension, indices)] * scaling_factor[channel_idx];
    channel_mse[channel_idx] += diff * diff;
  };

  std::vector<float> scaling_factor_base = scaling_factor;
  std::vector<std::pair<float, float>> golden_start_end(min.size());

  for (size_t i = 0; i < max_scale.size(); ++i)
  {
    golden_start_end[i].first = scaling_factor_base[i] * (1.0 - kRangeCoefficient);
    golden_start_end[i].second = scaling_factor_base[i] * (1.0 + kRangeCoefficient);
  }

  for (size_t i = 0; i < kSearchIterations; ++i)
  {
    for (size_t j = 0; j < scaling_factor.size(); ++j)
    {
      scaling_factor[j] = golden_start_end[j].second -
                          (golden_start_end[j].second - golden_start_end[j].first) / kPhi;
    }
    for (auto &val : channel_mse)
    {
      val = 0;
    }
    iterate_per_channel(node, channel_dim_index, quantize);
    iterate_per_channel(node, channel_dim_index, calculate_mse);
    auto channel_mse_x1 = channel_mse;

    for (size_t j = 0; j < scaling_factor.size(); ++j)
    {
      scaling_factor[j] =
        golden_start_end[j].first + (golden_start_end[j].second - golden_start_end[j].first) / kPhi;
    }
    for (auto &val : channel_mse)
    {
      val = 0;
    }
    iterate_per_channel(node, channel_dim_index, quantize);
    iterate_per_channel(node, channel_dim_index, calculate_mse);
    auto channel_mse_x2 = channel_mse;

    for (size_t k = 0; k < channel_mse_x1.size(); ++k)
    {
      if (channel_mse_x1[k] > channel_mse_x2[k])
      {
        golden_start_end[k].first = golden_start_end[k].second -
                                    (golden_start_end[k].second - golden_start_end[k].first) / kPhi;
      }
      else
      {
        golden_start_end[k].second =
          golden_start_end[k].first +
          (golden_start_end[k].second - golden_start_end[k].first) / kPhi;
      }
    }
  }
  for (size_t i = 0; i < golden_start_end.size(); ++i)
  {
    scaling_factor[i] = (golden_start_end[i].first + golden_start_end[i].second) / 2;
  }
  iterate_per_channel(node, channel_dim_index, quantize);
  iterate_per_channel(node, channel_dim_index, calculate_mse);
  auto channel_mse_opt = channel_mse;

  scaling_factor = scaling_factor_base;
  iterate_per_channel(node, channel_dim_index, quantize);
  iterate_per_channel(node, channel_dim_index, calculate_mse);
  auto channel_mse_base = channel_mse;

  // Checking if found scale is better than base
  for (size_t i = 0; i < channel_mse_base.size(); ++i)
  {
    if (channel_mse_opt[i] < channel_mse_base[i])
      scaling_factor[i] = (golden_start_end[i].first + golden_start_end[i].second) / 2;
    else
      channel_mse_opt[i] = channel_mse_base[i];
  }
  iterate_per_channel(node, channel_dim_index, quantize);

  node->dtype(out_type);      // change the type of tensor
  node->size<out_type>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<out_type>(i) = std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

// TODO Reduce duplicate code with QuantizeDequantizeWeights
template <loco::DataType out_type>
void sym_wquant_per_channel(CircleConst *node, std::vector<float> &min, std::vector<float> &max,
                            std::vector<float> &scaling_factor, std::vector<float> &nudged_min,
                            std::vector<float> &nudged_max, int32_t &channel_dim_index)
{
  assert(node->dtype() == loco::DataType::FLOAT32);
  assert(out_type == loco::DataType::S4 || out_type == loco::DataType::S8 ||
         out_type == loco::DataType::S16);

  const int32_t kMaxScale = max_for_sym_quant(out_type);
  const int32_t kMinScale = -kMaxScale;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(size);

  for (size_t i = 0; i < min.size(); ++i)
  {
    compute_sym_scale(min[i], max[i], scaling_factor[i], nudged_min[i], nudged_max[i], out_type);
  }

  auto quantize = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    const float scaling_factor_inv = 1.0 / scaling_factor[channel_idx];
    auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
    data = data < nudged_min[channel_idx] ? nudged_min[channel_idx] : data;
    data = data > nudged_max[channel_idx] ? nudged_max[channel_idx] : data;
    quantized_values[cal_offset(dimension, indices)] =
      static_cast<int32_t>(std::round(data * scaling_factor_inv));
  };

  iterate_per_channel(node, channel_dim_index, quantize);

  node->dtype(out_type);      // change the type of tensor
  node->size<out_type>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<out_type>(i) = std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void cal_minmax_per_channel(CircleConst *node, std::vector<float> &min, std::vector<float> &max,
                            int32_t &channel_dim_index)
{
  loco::TensorShape dimension;
  dimension.rank(4);

  if (!get_channel_dim_index(node, dimension, channel_dim_index))
  {
    throw std::runtime_error("Failed to find channel index in " + node->name());
  }
  auto size = dimension.dim(channel_dim_index).value();

  std::vector<bool> has_min_max_value(size, false);
  min.resize(size);
  max.resize(size);

  auto cal_minmax = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
    if (has_min_max_value[channel_idx])
    {
      min[channel_idx] = data < min[channel_idx] ? data : min[channel_idx];
      max[channel_idx] = data > max[channel_idx] ? data : max[channel_idx];
    }
    else
    {
      min[channel_idx] = data;
      max[channel_idx] = data;
      has_min_max_value[channel_idx] = true;
    }
  };

  iterate_per_channel(node, channel_dim_index, cal_minmax);
}

} // namespace

namespace luci
{

void QuantizeWeightsOnly::quantize_weights(luci::CircleConst *weights)
{
  // Find min/max per channel-wise
  if (granularity == QuantizationGranularity::ChannelWise)
  {
    auto quantparam = weights->quantparam();
    if (quantparam == nullptr)
    {
      // Find min/max on the fly
      // NOTE This is for the case when QuantizeDequantizeWeights is skipped
      // TODO Reduce duplicate codes
      std::vector<float> min;
      std::vector<float> max;
      int32_t channel_dim_index = 0;

      cal_minmax_per_channel(weights, min, max, channel_dim_index);

      std::vector<float> nudged_min(min.size());
      std::vector<float> nudged_max(min.size());
      std::vector<float> scaling_factor(min.size());
      std::vector<int64_t> zp(min.size());

      if (output_type == loco::DataType::S4)
      {
        switch (algorithm_params.type)
        {
          case luci::QuantizationAlgorithmType::MinimumMSE:
            sym_wquant_per_channel_minimum_mse<loco::DataType::S4>(
              weights, min, max, scaling_factor, nudged_min, nudged_max, channel_dim_index,
              algorithm_params);
            break;
          default:
            sym_wquant_per_channel<loco::DataType::S4>(weights, min, max, scaling_factor,
                                                       nudged_min, nudged_max, channel_dim_index);
            break;
        }
      }
      else if (output_type == loco::DataType::S8)
      {
        switch (algorithm_params.type)
        {
          case luci::QuantizationAlgorithmType::MinimumMSE:
            sym_wquant_per_channel_minimum_mse<loco::DataType::S8>(
              weights, min, max, scaling_factor, nudged_min, nudged_max, channel_dim_index,
              algorithm_params);
            break;
          default:
            sym_wquant_per_channel<loco::DataType::S8>(weights, min, max, scaling_factor,
                                                       nudged_min, nudged_max, channel_dim_index);
            break;
        }
      }
      else if (output_type == loco::DataType::S16)
      {
        switch (algorithm_params.type)
        {
          case luci::QuantizationAlgorithmType::MinimumMSE:
            sym_wquant_per_channel_minimum_mse<loco::DataType::S16>(
              weights, min, max, scaling_factor, nudged_min, nudged_max, channel_dim_index,
              algorithm_params);
            break;
          default:
            sym_wquant_per_channel<loco::DataType::S16>(weights, min, max, scaling_factor,
                                                        nudged_min, nudged_max, channel_dim_index);
            break;
        }
      }
      else
      {
        throw std::runtime_error("Weights-only quantization supports s8 and s16");
      }

      auto quantparam = std::make_unique<CircleQuantParam>();
      quantparam->scale = scaling_factor;
      quantparam->zerop = zp;
      quantparam->quantized_dimension = channel_dim_index;
      weights->quantparam(std::move(quantparam));

      return;
    }
  }
  else
    throw std::runtime_error("Weights-only quantization does not support layer-wise");
}

void QuantizeWeightsOnly::visit(luci::CircleConv2D *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeightsOnly visits node: " << node->name() << std::endl;

  auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
  if (!is_quantized(weights))
  {
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);
    quantize_weights(new_weights);
  }
}

void QuantizeWeightsOnly::visit(luci::CircleFullyConnected *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeightsOnly visit node: " << node->name() << std::endl;

  auto weights = loco::must_cast<luci::CircleConst *>(node->weights());
  if (!is_quantized(weights))
  {
    auto new_weights = luci::clone(weights);
    node->weights(new_weights);
    quantize_weights(new_weights);
  }
}

void QuantizeWeightsOnly::visit(luci::CircleDepthwiseConv2D *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeightsOnly visits node: " << node->name() << std::endl;

  auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
  if (!is_quantized(weights))
  {
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);
    quantize_weights(new_weights);
  }
}

void QuantizeWeightsOnly::visit(luci::CircleNode *) {}

} // namespace luci
