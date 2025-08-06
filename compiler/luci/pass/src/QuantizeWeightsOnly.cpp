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
#include <limits>

using namespace luci;

namespace
{

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
        sym_wquant_per_channel<loco::DataType::S4>(weights, min, max, scaling_factor, nudged_min,
                                                   nudged_max, channel_dim_index);
      }
      else if (output_type == loco::DataType::S8)
      {
        sym_wquant_per_channel<loco::DataType::S8>(weights, min, max, scaling_factor, nudged_min,
                                                   nudged_max, channel_dim_index);
      }
      else if (output_type == loco::DataType::S16)
      {
        sym_wquant_per_channel<loco::DataType::S16>(weights, min, max, scaling_factor, nudged_min,
                                                    nudged_max, channel_dim_index);
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
