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

#include "QuantizeWeights.h"
#include "QuantizationUtils.h"

#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>

#include <cmath>
#include <vector>
#include <functional>

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

void asym_wquant_per_channel(CircleConst *node, std::vector<float> &min,
                             std::vector<float> &scaling_factor, int32_t &channel_dim_index)
{
  assert(node->dtype() == loco::DataType::FLOAT32);

  const int32_t kMinScale = 0;
  const int32_t kMaxScale = 255;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(size);

  auto quantize = [&](uint32_t *indices, loco::TensorShape &dimension, int32_t channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    const float scaling_factor_inv = 1.0 / scaling_factor[channel_idx];
    auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
    quantized_values[cal_offset(dimension, indices)] =
      static_cast<int32_t>(std::round((data - min[channel_idx]) * scaling_factor_inv));
  };

  iterate_per_channel(node, channel_dim_index, quantize);

  node->dtype(loco::DataType::U8);      // change the type of tensor
  node->size<loco::DataType::U8>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::U8>(i) = std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void sym_wquant_per_channel(CircleConst *node, std::vector<float> &scaling_factor,
                            int32_t &channel_dim_index)
{
  assert(node->dtype() == loco::DataType::FLOAT32);

  const int32_t kMaxScale = std::numeric_limits<int16_t>::max();
  const int32_t kMinScale = -kMaxScale;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(size);

  auto quantize = [&](uint32_t *indices, loco::TensorShape &dimension, int32_t channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    const float scaling_factor_inv = 1.0 / scaling_factor[channel_idx];
    auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
    quantized_values[cal_offset(dimension, indices)] =
      static_cast<int32_t>(std::round(data * scaling_factor_inv));
  };

  iterate_per_channel(node, channel_dim_index, quantize);

  node->dtype(loco::DataType::S16);      // change the type of tensor
  node->size<loco::DataType::S16>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::S16>(i) =
      std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void asym_wquant_per_layer(CircleConst *node, float min, float scaling_factor)
{
  const int32_t kMinScale = 0;
  const int32_t kMaxScale = 255;

  uint32_t size = node->size<loco::DataType::FLOAT32>();

  const float scaling_factor_inv = 1.0 / scaling_factor;
  std::vector<int32_t> quantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    auto data = node->at<loco::DataType::FLOAT32>(i);
    quantized_values[i] = static_cast<int32_t>(std::round((data - min) * scaling_factor_inv));
  }

  node->dtype(loco::DataType::U8);      // change the type of tensor
  node->size<loco::DataType::U8>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::U8>(i) = std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

// Quantize const per channel
//
// The last dimension of const is the same as the dimension of channel
// And the rest of the const dimensions should be 1
// So, a 'single value' is quantized per channel
//
// Quantization spec (f: fp value, q: quantized value)
//
// uint8
//   Positive f: f = f * (q - 0) [q = 1, scale = f, zp = 0]
//   Negative f: f = (-f) * (q - 1) [q = 0, scale = -f, zp = 1]
//
// int16
//   Positive f: f = f * (q - 0) [q = 1, scale = f, zp = 0]
//   Negative f: f = (-f) * (q - 0) [q = -1, scale = -f, zp = 0]
void quant_const_per_channel(CircleConst *node, loco::DataType quant_type)
{
  assert(node->dtype() == loco::DataType::FLOAT32);
  assert(node->rank() > 0);

  for (uint32_t i = 0; i < node->rank() - 1; i++)
  {
    // Caller should call this function when the below condition is satisfied
    if (node->dim(i).value() != 1)
      throw std::runtime_error("Non-channel dimension of const node must be 1");
  }

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  assert(size == node->dim(node->rank() - 1).value());

  auto quantparam = std::make_unique<CircleQuantParam>();
  quantparam->quantized_dimension = node->rank() - 1;
  std::vector<int32_t> quantized_data(size);

  for (uint32_t i = 0; i < size; ++i)
  {
    auto data = node->at<loco::DataType::FLOAT32>(i);
    if (quant_type == loco::DataType::U8)
    {
      if (data >= 0)
      {
        quantparam->scale.push_back(data);
        quantparam->zerop.push_back(0);
        quantized_data[i] = 1;
      }
      else
      {
        quantparam->scale.push_back(-data);
        quantparam->zerop.push_back(1);
        quantized_data[i] = 0;
      }
    }
    else if (quant_type == loco::DataType::S16)
    {
      if (data >= 0)
      {
        quantparam->scale.push_back(data);
        quantized_data[i] = 1;
      }
      else
      {
        quantparam->scale.push_back(-data);
        quantized_data[i] = -1;
      }
      quantparam->zerop.push_back(0);
    }
  }
  node->quantparam(std::move(quantparam));

  switch (quant_type)
  {
    case loco::DataType::U8:
      node->dtype(loco::DataType::U8);
      node->size<loco::DataType::U8>(size);
      for (uint32_t i = 0; i < size; ++i)
      {
        assert(quantized_data[i] == 0 || quantized_data[i] == 1);
        node->at<loco::DataType::U8>(i) = quantized_data[i];
      }
      break;
    case loco::DataType::S16:
      node->dtype(loco::DataType::S16);
      node->size<loco::DataType::S16>(size);
      for (uint32_t i = 0; i < size; ++i)
      {
        assert(quantized_data[i] == -1 || quantized_data[i] == 1);
        node->at<loco::DataType::S16>(i) = quantized_data[i];
      }
      break;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

} // namespace

namespace luci
{

void QuantizeWeights::quantize_weights(luci::CircleConst *weights)
{
  // Find min/max per channel-wise
  if (granularity == QuantizationGranularity::ChannelWise)
  {
    auto quantparam = weights->quantparam();
    if (quantparam == nullptr)
    {
      assert(false && "quantparam is nullptr");
      return;
    }

    auto min = quantparam->min;
    auto scaling_factor = quantparam->scale;
    int32_t channel_dim_index = 0;

    if (output_type == loco::DataType::U8)
    {
      asym_wquant_per_channel(weights, min, scaling_factor, channel_dim_index);
    }
    else
    {
      sym_wquant_per_channel(weights, scaling_factor, channel_dim_index);
    }
    quantparam->min.clear();
    quantparam->max.clear();
    quantparam->quantized_dimension = channel_dim_index;
  }
  // Find min/max per layer-wise
  else
  {
    // Quantize using recorded quantparam
    auto quantparam = weights->quantparam();
    assert(quantparam != nullptr);
    assert(quantparam->min.size() == 1);   // only support layer-wise quant
    assert(quantparam->scale.size() == 1); // only support layer-wise quant
    auto min = quantparam->min[0];
    auto scaling_factor = quantparam->scale[0];
    asym_wquant_per_layer(weights, min, scaling_factor);
    quantparam->min.clear();
    quantparam->max.clear();
  }
}
void QuantizeWeights::visit(luci::CircleConv2D *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeights QuantizeWeights::visit node: " << node->name() << std::endl;

  auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
  if (!is_quantized(weights))
  {
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);
    quantize_weights(new_weights);
  }
}

void QuantizeWeights::visit(luci::CircleDepthwiseConv2D *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeights QuantizeWeights::visit node: " << node->name() << std::endl;

  auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
  if (!is_quantized(weights))
  {
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);
    quantize_weights(new_weights);
  }
}

void QuantizeWeights::visit(luci::CircleInstanceNorm *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeights QuantizeWeights::visit node: " << node->name() << std::endl;

  auto gamma = loco::must_cast<luci::CircleConst *>(node->gamma());
  auto beta = loco::must_cast<luci::CircleConst *>(node->beta());

  if (!is_quantized(gamma))
  {
    assert(gamma->dtype() == loco::DataType::FLOAT32);
    auto new_gamma = luci::clone(gamma);
    if (granularity == QuantizationGranularity::LayerWise)
      quant_const(new_gamma, output_type);
    else if (granularity == QuantizationGranularity::ChannelWise)
      quant_const_per_channel(new_gamma, output_type);
    node->gamma(new_gamma);
  }
  if (!is_quantized(beta))
  {
    assert(beta->dtype() == loco::DataType::FLOAT32);
    auto new_beta = luci::clone(beta);
    if (granularity == QuantizationGranularity::LayerWise)
      quant_const(new_beta, output_type);
    else if (granularity == QuantizationGranularity::ChannelWise)
      quant_const_per_channel(new_beta, output_type);
    node->beta(new_beta);
  }
}

void QuantizeWeights::visit(luci::CirclePRelu *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeights QuantizeWeights::visit node: " << node->name() << std::endl;

  auto alpha = loco::must_cast<luci::CircleConst *>(node->alpha());

  if (!is_quantized(alpha))
  {
    assert(alpha->dtype() == loco::DataType::FLOAT32);
    auto new_alpha = luci::clone(alpha);
    if (granularity == QuantizationGranularity::LayerWise)
      quant_const(new_alpha, output_type);
    else if (granularity == QuantizationGranularity::ChannelWise)
      quant_const_per_channel(new_alpha, output_type);
    node->alpha(new_alpha);
  }
}

void QuantizeWeights::visit(luci::CircleTransposeConv *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeights QuantizeWeights::visit node: " << node->name() << std::endl;

  auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
  if (!is_quantized(weights))
  {
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);
    quantize_weights(new_weights);
  }
}

void QuantizeWeights::visit(luci::CircleFullyConnected *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeights QuantizeWeights::visit node: " << node->name() << std::endl;

  auto weights = loco::must_cast<luci::CircleConst *>(node->weights());
  if (!is_quantized(weights))
  {
    auto new_weights = luci::clone(weights);
    node->weights(new_weights);
    quantize_weights(new_weights);
  }
}

void QuantizeWeights::visit(luci::CircleNode *) {}

} // namespace luci
