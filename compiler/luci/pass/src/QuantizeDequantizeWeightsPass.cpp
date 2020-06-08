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
    if (conv != nullptr && conv->filter() == circle_const)
      return true;

    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    if (dw_conv != nullptr && dw_conv->filter() == circle_const)
      return true;

    auto tw_conv = dynamic_cast<CircleTransposeConv *>(out);
    if (tw_conv != nullptr && tw_conv->filter() == circle_const)
      return true;

    auto fc = dynamic_cast<CircleFullyConnected *>(out);
    if (fc != nullptr && fc->weights() == circle_const)
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
  QuantizeDequantizeWeights(loco::DataType input, loco::DataType output)
      : input_type(input), output_type(output)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;

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

        // Find min/max per layer-wise
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();
        for (uint32_t i = 0; i < circle_const->size<loco::DataType::FLOAT32>(); i++)
        {
          auto data = circle_const->at<loco::DataType::FLOAT32>(i);
          min = data < min ? data : min;
          max = data > max ? data : max;
        }
        float scaling_factor;
        int64_t zp;
        float nudged_min;
        float nudged_max;

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
    QuantizeDequantizeWeights qw(_input_dtype, _output_dtype);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qw);
  }

  INFO(l) << "QuantizeDequantizeWeightsPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
