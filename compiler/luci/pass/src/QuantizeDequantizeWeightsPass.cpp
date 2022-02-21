/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "luci/Pass/QuantizeDequantizeWeightsPass.h"
#include "QuantizationUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>
#include <loco/IR/TensorShape.h>

#include <iostream>
#include <cmath>
#include <functional>

namespace
{

using namespace luci;
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

} // namespace

namespace luci
{

namespace
{

void cal_minmax_per_channel(CircleConst *node, std::vector<float> &min, std::vector<float> &max)
{
  loco::TensorShape dimension;
  dimension.rank(4);
  int32_t channel_dim_index{0};

  if (!get_channel_dim_index(node, dimension, channel_dim_index))
  {
    assert(false);
    return;
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

  iterate_per_channel(node, cal_minmax);
}

void sym_wquant_per_channel(CircleConst *node, std::vector<float> &min, std::vector<float> &max,
                            std::vector<float> &scaling_factor, std::vector<int64_t> &zp,
                            std::vector<float> &nudged_min, std::vector<float> &nudged_max)
{
  assert(node->dtype() == loco::DataType::FLOAT32);
  const int32_t kMaxScale = std::numeric_limits<int16_t>::max();
  const int32_t kMinScale = -kMaxScale;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(size);

  for (size_t i = 0; i < min.size(); ++i)
  {
    compute_sym_scale_zp(min[i], max[i], scaling_factor[i], zp[i], nudged_min[i], nudged_max[i]);
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

  iterate_per_channel(node, quantize);

  node->dtype(loco::DataType::S16);      // change the type of tensor
  node->size<loco::DataType::S16>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::S16>(i) =
      std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void sym_wdequant_per_channel(CircleConst *node, std::vector<float> &scaling_factor)
{
  assert(node->dtype() == loco::DataType::S16);
  uint32_t size = node->size<loco::DataType::S16>();
  std::vector<float> dequantized_values(size);

  auto dequantize = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    auto data = node->at<loco::DataType::S16>(cal_offset(dimension, indices));
    dequantized_values[cal_offset(dimension, indices)] =
      static_cast<float>(data) * scaling_factor[channel_idx];
  };

  iterate_per_channel(node, dequantize);

  node->dtype(loco::DataType::FLOAT32);      // change the type of tensor
  node->size<loco::DataType::FLOAT32>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::FLOAT32>(i) = dequantized_values[i];
  }
}

void asymmetric_wquant_per_channel(CircleConst *node, std::vector<float> &min,
                                   std::vector<float> &max, std::vector<float> &scaling_factor,
                                   std::vector<int64_t> &zp, std::vector<float> &nudged_min,
                                   std::vector<float> &nudged_max)
{
  assert(node->dtype() == loco::DataType::FLOAT32);

  const int32_t kMinScale = 0;
  const int32_t kMaxScale = 255;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(size);

  for (size_t i = 0; i < min.size(); ++i)
  {
    compute_asym_scale_zp(min[i], max[i], scaling_factor[i], zp[i], nudged_min[i], nudged_max[i]);
  }

  auto quantize = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    const float scaling_factor_inv = 1.0 / scaling_factor[channel_idx];
    auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
    data = data < nudged_min[channel_idx] ? nudged_min[channel_idx] : data;
    data = data > nudged_max[channel_idx] ? nudged_max[channel_idx] : data;
    quantized_values[cal_offset(dimension, indices)] =
      static_cast<int32_t>(std::round((data - nudged_min[channel_idx]) * scaling_factor_inv));
  };

  iterate_per_channel(node, quantize);

  node->dtype(loco::DataType::U8);      // change the type of tensor
  node->size<loco::DataType::U8>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
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

  auto dequantize = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    auto data = node->at<loco::DataType::U8>(cal_offset(dimension, indices));
    dequantized_values[cal_offset(dimension, indices)] =
      static_cast<float>(data) * scaling_factor[channel_idx] + nudged_min[channel_idx];
  };

  iterate_per_channel(node, dequantize);

  node->dtype(loco::DataType::FLOAT32);      // change the type of tensor
  node->size<loco::DataType::FLOAT32>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::FLOAT32>(i) = dequantized_values[i];
  }
}

void asymmetric_wdequant_with_minmax_per_layer(CircleConst *node, float scaling_factor,
                                               float nudged_min)
{
  uint32_t size = node->size<loco::DataType::U8>();
  std::vector<float> dequantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    auto data = node->at<loco::DataType::U8>(i);
    dequantized_values[i] = static_cast<float>(data) * scaling_factor + nudged_min;
  }

  node->dtype(loco::DataType::FLOAT32);      // change the type of tensor
  node->size<loco::DataType::FLOAT32>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::FLOAT32>(i) = dequantized_values[i];
  }
}

/**
 * @brief QuantizeDequantizeWeights quantizes and dequantizes tensors for weights
 * @details Find min/max values on the fly, quantize the model, and dequantize the model
 */
struct QuantizeDequantizeWeights final : public luci::CircleNodeMutableVisitor<bool>
{
  QuantizeDequantizeWeights(loco::DataType input, loco::DataType output,
                            QuantizationGranularity granularity)
    : input_type(input), output_type(output), granularity(granularity)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;
  QuantizationGranularity granularity;

private:
  // Fake quantize weights (Only u8 quantization is supported for LWQ)
  void fake_quantize_lwq(luci::CircleConst *weights) const
  {
    assert(output_type == loco::DataType::U8); // FIX_CALLER_UNLESS

    // Find min/max per layer
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();
    for (uint32_t i = 0; i < weights->size<loco::DataType::FLOAT32>(); i++)
    {
      auto data = weights->at<loco::DataType::FLOAT32>(i);
      min = data < min ? data : min;
      max = data > max ? data : max;
    }
    float scaling_factor{0};
    int64_t zp{0};
    float nudged_min{0};
    float nudged_max{0};

    asymmetric_wquant_with_minmax_per_layer(weights, min, max, scaling_factor, zp, nudged_min,
                                            nudged_max);
    asymmetric_wdequant_with_minmax_per_layer(weights, scaling_factor, nudged_min);
    auto quantparam = std::make_unique<CircleQuantParam>();
    quantparam->min.push_back(nudged_min);
    quantparam->max.push_back(nudged_max);
    quantparam->scale.push_back(scaling_factor);
    quantparam->zerop.push_back(zp);
    weights->quantparam(std::move(quantparam));
  }

private:
  // Fake quantize weights (u8/s16 quantization are supported for CWQ)
  void fake_quantize_cwq(luci::CircleConst *weights) const
  {
    assert(output_type == loco::DataType::U8 ||
           output_type == loco::DataType::S16); // FIX_CALLER_UNLESS

    // Find min/max per channel
    std::vector<float> min;
    std::vector<float> max;

    cal_minmax_per_channel(weights, min, max);

    std::vector<float> nudged_min(min.size());
    std::vector<float> nudged_max(min.size());
    std::vector<float> scaling_factor(min.size());
    std::vector<int64_t> zp(min.size());

    if (output_type == loco::DataType::U8)
    {
      asymmetric_wquant_per_channel(weights, min, max, scaling_factor, zp, nudged_min, nudged_max);
      asymmetric_wdequant_per_channel(weights, scaling_factor, nudged_min);
    }
    else
    {
      sym_wquant_per_channel(weights, min, max, scaling_factor, zp, nudged_min, nudged_max);
      sym_wdequant_per_channel(weights, scaling_factor);
    }

    auto quantparam = std::make_unique<CircleQuantParam>();
    quantparam->min = nudged_min;
    quantparam->max = nudged_max;
    quantparam->scale = scaling_factor;
    quantparam->zerop = zp;
    weights->quantparam(std::move(quantparam));
  }

private:
  void fake_quantize(luci::CircleConst *weights) const
  {
    switch (granularity)
    {
      case luci::QuantizationGranularity::ChannelWise:
        fake_quantize_cwq(weights);
        break;
      case luci::QuantizationGranularity::LayerWise:
        fake_quantize_lwq(weights);
        break;
      default:
        throw std::invalid_argument("Unsupported granularity");
    }
  }

private:
  // Check if
  // 1. node is const
  // 2. node was not quantized
  bool is_quantizable(loco::Node *node)
  {
    auto const_node = dynamic_cast<luci::CircleConst *>(node);
    if (not const_node)
      return false;

    // Skip if this is already quantized
    if (is_quantized(const_node))
      return false;

    return true;
  }

  bool visit(luci::CircleNode *node)
  {
    assert(output_type == loco::DataType::U8 || output_type == loco::DataType::S16);
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeights visit node: " << node->name() << std::endl;

    if (not is_quantizable(node))
      return false;

    if (not is_weights(node))
      return false;

    auto circle_const = loco::must_cast<luci::CircleConst *>(node);
    fake_quantize(circle_const);
    return false;
  }
};

} // namespace

bool QuantizeDequantizeWeightsPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "QuantizeDequantizeWeightsPass Start" << std::endl;

  // Quantize weights
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    QuantizeDequantizeWeights qw(_input_model_dtype, _output_model_dtype, _granularity);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qw);
  }

  INFO(l) << "QuantizeDequantizeWeightsPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
