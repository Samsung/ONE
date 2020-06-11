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

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>
#include <loco/IR/TensorShape.h>

#include <iostream>
#include <cmath>

namespace luci
{

namespace
{

void compute_asym_scale_zp(float min, float max, float &scaling_factor, int64_t &zp,
                           float &nudged_min, float &nudged_max)
{
  assert(min != max);

  const int32_t kMinScale = 0;
  const int32_t kMaxScale = 255;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const double rmin = std::fmin(0, min);
  const double rmax = std::fmax(0, max);

  double scale = (rmax - rmin) / (qmax_double - qmin_double);
  const double zero_point_from_min = qmin_double - rmin / scale;
  const double zero_point_from_max = qmax_double - rmax / scale;
  const double zero_point_from_min_error = std::abs(qmin_double) + std::abs(rmin / scale);
  const double zero_point_from_max_error = std::abs(qmax_double) + std::abs(rmax / scale);
  const double zero_point_double = zero_point_from_min_error < zero_point_from_max_error
                                       ? zero_point_from_min
                                       : zero_point_from_max;
  uint8_t nudged_zero_point = 0;
  if (zero_point_double <= qmin_double)
  {
    nudged_zero_point = kMinScale;
  }
  else if (zero_point_double >= qmax_double)
  {
    nudged_zero_point = kMaxScale;
  }
  else
  {
    nudged_zero_point = static_cast<uint8_t>(std::round(zero_point_double));
  }

  nudged_min = static_cast<float>((qmin_double - nudged_zero_point) * scale);
  nudged_max = static_cast<float>((qmax_double - nudged_zero_point) * scale);

  scaling_factor = scale;
  zp = nudged_zero_point;
}

bool get_channel_dim_index(CircleConst *node, loco::TensorShape &dimension, int &channel_dim_index)
{
  auto succs = loco::succs(node);
  if (succs.size() != 1) // assume weights is used by only one node
    return false;

  for (auto out : succs)
  {
    auto conv = dynamic_cast<CircleConv2D *>(out);
    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    auto tw_conv = dynamic_cast<CircleTransposeConv *>(out);
    auto fc = dynamic_cast<CircleFullyConnected *>(out);

    if (conv != nullptr && conv->filter() == node)
    {
      assert(node->rank() == 4);
      dimension.dim(0).set(node->dim(0).value());
      dimension.dim(1).set(node->dim(1).value());
      dimension.dim(2).set(node->dim(2).value());
      dimension.dim(3).set(node->dim(3).value());
      channel_dim_index = 3;
      return true;
    }
    else if ((tw_conv != nullptr && tw_conv->filter() == node) ||
             (dw_conv != nullptr && dw_conv->filter() == node))
    {
      assert(node->rank() == 4);
      dimension.dim(0).set(node->dim(0).value());
      dimension.dim(1).set(node->dim(1).value());
      dimension.dim(2).set(node->dim(2).value());
      dimension.dim(3).set(node->dim(3).value());
      channel_dim_index = 2;
      return true;
    }
    else if (fc != nullptr && fc->weights() == node)
    {
      assert(node->rank() == 2);
      dimension.dim(0).set(1);
      dimension.dim(1).set(1);
      dimension.dim(2).set(node->dim(0).value());
      dimension.dim(3).set(node->dim(1).value());
      channel_dim_index = 3;
      return true;
    }
    else
    {
      // node does not support channle-wise quantization
      assert(false);
    }
  }

  return false;
}

uint32_t cal_offset(loco::TensorShape &dimension, uint32_t *indices)
{
  return indices[0] * dimension.dim(1).value() * dimension.dim(2).value() *
             dimension.dim(3).value() +
         indices[1] * dimension.dim(2).value() * dimension.dim(3).value() +
         indices[2] * dimension.dim(3).value() + indices[3];
}

void cal_minmax_per_channel(CircleConst *node, std::vector<float> &min, std::vector<float> &max)
{
  loco::TensorShape dimension;
  dimension.rank(4);
  uint32_t indices[4] = {
      0,
  };
  int channel_dim_index{0};
  int size{0};

  bool ret = get_channel_dim_index(node, dimension, channel_dim_index);
  assert(ret);
  size = dimension.dim(channel_dim_index).value();

  std::vector<bool> has_min_max_value(size, false);
  min.resize(size);
  max.resize(size);
  for (indices[0] = 0; indices[0] < dimension.dim(0).value(); indices[0]++)
  {
    for (indices[1] = 0; indices[1] < dimension.dim(1).value(); indices[1]++)
    {
      for (indices[2] = 0; indices[2] < dimension.dim(2).value(); indices[2]++)
      {
        for (indices[3] = 0; indices[3] < dimension.dim(3).value(); indices[3]++)
        {
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
        }
      }
    }
  }
}

void wquant_per_channel(CircleConst *node, std::vector<float> &min, std::vector<float> &max,
                        std::vector<float> &scaling_factor, std::vector<int64_t> &zp,
                        std::vector<float> &nudged_min, std::vector<float> &nudged_max)
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

  loco::TensorShape dimension;
  dimension.rank(4);
  uint32_t indices[4] = {
      0,
  };
  int channel_dim_index{0};

  bool ret = get_channel_dim_index(node, dimension, channel_dim_index);
  assert(ret);

  for (indices[0] = 0; indices[0] < dimension.dim(0).value(); indices[0]++)
  {
    for (indices[1] = 0; indices[1] < dimension.dim(1).value(); indices[1]++)
    {
      for (indices[2] = 0; indices[2] < dimension.dim(2).value(); indices[2]++)
      {
        for (indices[3] = 0; indices[3] < dimension.dim(3).value(); indices[3]++)
        {
          int channel_idx = indices[channel_dim_index];
          const float scaling_factor_inv = 1.0 / scaling_factor[channel_idx];
          auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
          data = data < nudged_min[channel_idx] ? nudged_min[channel_idx] : data;
          data = data > nudged_max[channel_idx] ? nudged_max[channel_idx] : data;
          quantized_values[cal_offset(dimension, indices)] = static_cast<int32_t>(
              std::round((data - nudged_min[channel_idx]) * scaling_factor_inv));
        }
      }
    }
  }

  node->dtype(loco::DataType::U8);      // change the type of tensor
  node->size<loco::DataType::U8>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::U8>(i) = std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void wdequant_per_channel(CircleConst *node, std::vector<float> &scaling_factor,
                          std::vector<float> &nudged_min)
{
  assert(node->dtype() == loco::DataType::U8);
  uint32_t size = node->size<loco::DataType::U8>();
  std::vector<float> dequantized_values(size);

  loco::TensorShape dimension;
  dimension.rank(4);
  uint32_t indices[4] = {
      0,
  };
  int channel_dim_index{0};

  bool ret = get_channel_dim_index(node, dimension, channel_dim_index);
  assert(ret);

  for (indices[0] = 0; indices[0] < dimension.dim(0).value(); indices[0]++)
  {
    for (indices[1] = 0; indices[1] < dimension.dim(1).value(); indices[1]++)
    {
      for (indices[2] = 0; indices[2] < dimension.dim(2).value(); indices[2]++)
      {
        for (indices[3] = 0; indices[3] < dimension.dim(3).value(); indices[3]++)
        {
          int channel_idx = indices[channel_dim_index];
          auto data = node->at<loco::DataType::U8>(cal_offset(dimension, indices));
          dequantized_values[cal_offset(dimension, indices)] =
              static_cast<float>(data) * scaling_factor[channel_idx] + nudged_min[channel_idx];
        }
      }
    }
  }

  node->dtype(loco::DataType::FLOAT32);      // change the type of tensor
  node->size<loco::DataType::FLOAT32>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::FLOAT32>(i) = dequantized_values[i];
  }
}

void wquant_with_minmax_per_layer(CircleConst *node, float min, float max, float &scaling_factor,
                                  int64_t &zp, float &nudged_min, float &nudged_max)
{

  const int32_t kMinScale = 0;
  const int32_t kMaxScale = 255;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  if (min == max)
  {
    node->dtype(loco::DataType::U8);      // change the type of tensor
    node->size<loco::DataType::U8>(size); // resize tensor
    for (int i = 0; i < static_cast<int32_t>(size); ++i)
      node->at<loco::DataType::U8>(i) = 0;

    scaling_factor = 1;
    zp = 0;
    return;
  }

  compute_asym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
  const float scaling_factor_inv = 1.0 / scaling_factor;
  std::vector<int32_t> quantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    // clipping
    auto data = node->at<loco::DataType::FLOAT32>(i);
    data = data < nudged_min ? nudged_min : data;
    data = data > nudged_max ? nudged_max : data;
    quantized_values[i] =
        static_cast<int32_t>(std::round((data - nudged_min) * scaling_factor_inv));
  }

  node->dtype(loco::DataType::U8);      // change the type of tensor
  node->size<loco::DataType::U8>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::U8>(i) = std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void wdequant_with_minmax_per_layer(CircleConst *node, float scaling_factor, float nudged_min)
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

bool is_quantized(const CircleNode *node)
{
  return node->dtype() == loco::DataType::U8 || // activation, weight
         node->dtype() == loco::DataType::S32;  // bias
}

// Check if node is weights of conv2d, transepose_conv2d, depthwise_conv2d, or fully_connected layer
bool is_weights(CircleNode *node)
{
  auto circle_const = dynamic_cast<CircleConst *>(node);
  if (circle_const == nullptr)
    return false;

  auto succs = loco::succs(node);
  if (succs.size() != 1) // assume weights is used by only one node
    return false;

  for (auto out : succs)
  {
    auto conv = dynamic_cast<CircleConv2D *>(out);
    if (conv != nullptr && conv->filter() == circle_const && circle_const->rank() == 4)
      return true;

    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    if (dw_conv != nullptr && dw_conv->filter() == circle_const && circle_const->rank() == 4)
      return true;

    auto tw_conv = dynamic_cast<CircleTransposeConv *>(out);
    if (tw_conv != nullptr && tw_conv->filter() == circle_const && circle_const->rank() == 4)
      return true;

    auto fc = dynamic_cast<CircleFullyConnected *>(out);
    if (fc != nullptr && fc->weights() == circle_const && circle_const->rank() == 2)
      return true;
  }
  return false;
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

  // Quantize and dequantize input tensors of each node
  bool visit(luci::CircleNode *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeights visit node: " << node->name() << std::endl;
    auto arity = node->arity();
    for (uint32_t i = 0; i < arity; i++)
    {
      auto input_node = node->arg(i);
      auto circle_node = loco::must_cast<luci::CircleNode *>(input_node);

      // Check if this is already quantized
      if (is_quantized(circle_node))
        continue;

      if (is_weights(circle_node))
      {
        auto circle_const = loco::must_cast<luci::CircleConst *>(circle_node);

        // Find min/max per channel-wise
        if (granularity == QuantizationGranularity::ChannelWise)
        {
          std::vector<float> min;
          std::vector<float> max;

          cal_minmax_per_channel(circle_const, min, max);

          std::vector<float> nudged_min(min.size());
          std::vector<float> nudged_max(min.size());
          std::vector<float> scaling_factor(min.size());
          std::vector<int64_t> zp(min.size());

          wquant_per_channel(circle_const, min, max, scaling_factor, zp, nudged_min, nudged_max);
          wdequant_per_channel(circle_const, scaling_factor, nudged_min);

          auto quantparam = std::make_unique<CircleQuantParam>();
          quantparam->min = nudged_min;
          quantparam->max = nudged_max;
          quantparam->scale = scaling_factor;
          quantparam->zerop = zp;
          circle_node->quantparam(std::move(quantparam));
        }
        // Find min/max per layer-wise
        else
        {
          float min = std::numeric_limits<float>::max();
          float max = std::numeric_limits<float>::lowest();
          for (uint32_t i = 0; i < circle_const->size<loco::DataType::FLOAT32>(); i++)
          {
            auto data = circle_const->at<loco::DataType::FLOAT32>(i);
            min = data < min ? data : min;
            max = data > max ? data : max;
          }
          float scaling_factor{0};
          int64_t zp{0};
          float nudged_min{0};
          float nudged_max{0};

          wquant_with_minmax_per_layer(circle_const, min, max, scaling_factor, zp, nudged_min,
                                       nudged_max);
          wdequant_with_minmax_per_layer(circle_const, scaling_factor, nudged_min);
          auto quantparam = std::make_unique<CircleQuantParam>();
          quantparam->min.push_back(nudged_min);
          quantparam->max.push_back(nudged_max);
          quantparam->scale.push_back(scaling_factor);
          quantparam->zerop.push_back(zp);
          circle_node->quantparam(std::move(quantparam));
        }
      }
    }
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
    QuantizeDequantizeWeights qw(_input_dtype, _output_dtype, _granularity);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qw);
  }

  INFO(l) << "QuantizeDequantizeWeightsPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
