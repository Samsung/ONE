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

#include "luci/Pass/QuantizeWithMinMaxPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>

#include <oops/UserExn.h>

#include <iostream>
#include <cmath>

namespace luci
{

namespace
{

void compute_asym_scale_zp(float min, float max, float *scaling_factor, int64_t *zp)
{
  assert(min != max);
  if (min == max)
  {
    *scaling_factor = 1;
    *zp = 0;
  }
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
  int8_t nudged_zero_point = 0;
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
    nudged_zero_point = static_cast<int8_t>(std::round(zero_point_double));
  }
  *scaling_factor = scale;
  *zp = nudged_zero_point;
}

// Check if the node is the bias of Conv2D, DepthwiseConv2D, or FullyConnected layer
// If true, return <input, weight> pair of the successor node (used to quantize bias)
// If flase, return <nullptr, nullptr>
std::pair<loco::Node *, loco::Node *> get_input_weight_of_bias(CircleNode *node)
{
  auto circle_const = dynamic_cast<CircleConst *>(node);
  if (circle_const == nullptr)
    return std::make_pair(nullptr, nullptr);

  auto succs = loco::succs(node);
  if (succs.size() != 1) // assume bias is used by only one node
    return std::make_pair(nullptr, nullptr);

  for (auto out : succs)
  {
    auto conv = dynamic_cast<CircleConv2D *>(out);
    if (conv != nullptr && conv->bias() == circle_const)
    {
      assert(conv->input() != nullptr);
      assert(conv->filter() != nullptr);
      return std::make_pair(conv->input(), conv->filter());
    }
    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    if (dw_conv != nullptr && dw_conv->bias() == circle_const)
    {
      assert(dw_conv->input() != nullptr);
      assert(dw_conv->filter() != nullptr);
      return std::make_pair(dw_conv->input(), dw_conv->filter());
    }
    auto fc = dynamic_cast<CircleFullyConnected *>(out);
    if (fc != nullptr && fc->bias() == circle_const)
    {
      assert(fc->input() != nullptr);
      assert(fc->weights() != nullptr);
      return std::make_pair(fc->input(), fc->weights());
    }
  }
  return std::make_pair(nullptr, nullptr);
}

void asym_quant_bias(CircleConst *node, float input_scale, float weight_scale,
                     float *scaling_factor, int64_t *zp)
{
  float scale = input_scale * weight_scale;
  const float scaling_factor_inv = (scale == 0) ? 0 : 1.0 / scale;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(size);
  for (int i = 0; i < static_cast<int32_t>(size); ++i)
  {
    quantized_values[i] =
        static_cast<int32_t>(std::round(node->at<loco::DataType::FLOAT32>(i) * scaling_factor_inv));
  }

  node->dtype(loco::DataType::S32);      // change the type of tensor
  node->size<loco::DataType::S32>(size); // resize tensor
  const int32_t kMinScale = std::numeric_limits<int32_t>::lowest();
  const int32_t kMaxScale = std::numeric_limits<int32_t>::max();
  for (int i = 0; i < static_cast<int32_t>(size); ++i)
  {
    node->at<loco::DataType::S32>(i) =
        std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
  *scaling_factor = scale;
  *zp = 0;
}

bool has_min_max(const CircleNode *node)
{
  return node->quantparam() && !node->quantparam()->min.empty() && !node->quantparam()->max.empty();
}

bool is_quantized(const CircleNode *node)
{
  return node->dtype() == loco::DataType::U8 || // activation, weight
         node->dtype() == loco::DataType::S32;  // bias
}

void asym_wquant(CircleConst *node, float min, float scaling_factor)
{
  const int32_t kMinScale = 0;
  const int32_t kMaxScale = 255;

  uint32_t size = node->size<loco::DataType::FLOAT32>();

  const float scaling_factor_inv = 1.0 / scaling_factor;
  std::vector<int32_t> quantized_values(size);
  for (int i = 0; i < static_cast<int32_t>(size); ++i)
  {
    auto data = node->at<loco::DataType::FLOAT32>(i);
    quantized_values[i] = static_cast<int32_t>(std::round((data - min) * scaling_factor_inv));
  }

  node->dtype(loco::DataType::U8);      // change the type of tensor
  node->size<loco::DataType::U8>(size); // resize tensor
  for (int i = 0; i < static_cast<int32_t>(size); ++i)
  {
    node->at<loco::DataType::U8>(i) = std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

// Check if node is weights of conv2d, depthwise_conv2d, or fully_connected layer
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

    auto fc = dynamic_cast<CircleFullyConnected *>(out);
    if (fc != nullptr && fc->weights() == circle_const)
      return true;
  }
  return false;
}

/**
 * @brief QuantizeActivation quantizes tensors for activations
 * @details Quantize using recorded min/max values
 */
struct QuantizeActivation final : public luci::CircleNodeMutableVisitor<bool>
{
  QuantizeActivation(loco::DataType input, loco::DataType output)
      : input_type(input), output_type(output)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;

  // Quantize input tensors of each node
  bool visit(luci::CircleNode *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeActivation visit node: " << node->name() << std::endl;
    auto arity = node->arity();
    for (uint32_t i = 0; i < arity; i++)
    {
      auto input_node = node->arg(i);
      auto circle_node = loco::must_cast<luci::CircleNode *>(input_node);

      // Check if this is already quantized
      if (is_quantized(circle_node))
        continue;

      // Check if this is bias (bias is quantized later)
      auto iw = get_input_weight_of_bias(circle_node);
      if (iw.first != nullptr && iw.second != nullptr)
        continue;

      // Check if this is activation
      // We assume min/max are recorded only for activations
      if (has_min_max(circle_node))
      {
        // Quantize using recorded min/max
        auto quantparam = circle_node->quantparam();
        assert(quantparam->min.size() == 1); // only support layer-wise quant
        assert(quantparam->max.size() == 1); // only support layer-wise quant
        auto min = quantparam->min[0];
        auto max = quantparam->max[0];

        float scaling_factor;
        int64_t zp;
        compute_asym_scale_zp(min, max, &scaling_factor, &zp);
        circle_node->quantparam()->scale.push_back(scaling_factor);
        circle_node->quantparam()->zerop.push_back(zp);
        circle_node->dtype(loco::DataType::U8);
      }
    }
    return false;
  }
};

struct QuantizeBias final : public luci::CircleNodeMutableVisitor<bool>
{
  QuantizeBias(loco::DataType input, loco::DataType output, QuantizationGranularity gr)
      : input_type(input), output_type(output), granularity(gr)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;
  QuantizationGranularity granularity;

  // Quantize bias node
  bool visit(luci::CircleNode *node)
  {
    // Check if this is already quantized
    if (is_quantized(node))
      return false;

    // Check if this is bias
    auto iw = get_input_weight_of_bias(node);
    if (iw.first == nullptr || iw.second == nullptr)
      return false;

    auto input = loco::must_cast<luci::CircleNode *>(iw.first);
    auto weight = loco::must_cast<luci::CircleNode *>(iw.second);

    if (granularity == QuantizationGranularity::ChannelWise)
    {
      // will be implemented.
    }
    else
    {
      assert(input->quantparam()->scale.size() == 1); // Only support per-layer quant
      auto input_scale = input->quantparam()->scale[0];

      assert(weight->quantparam()->scale.size() == 1); // Only support per-layer quant
      auto weight_scale = weight->quantparam()->scale[0];

      auto circle_const = loco::must_cast<luci::CircleConst *>(node);
      float scaling_factor{0};
      int64_t zp{0};
      asym_quant_bias(circle_const, input_scale, weight_scale, &scaling_factor, &zp);
      auto quantparam = std::make_unique<CircleQuantParam>();
      quantparam->scale.push_back(scaling_factor);
      quantparam->zerop.push_back(zp);
      assert(circle_const->quantparam() == nullptr); // bias should not be quantized before
      circle_const->quantparam(std::move(quantparam));
    }
    return false;
  }
};

/**
 * @brief QuantizeWeights quantizes tensors for weights
 * @details Find min/max values on the fly and then quantize
 */
struct QuantizeWeights final : public luci::CircleNodeMutableVisitor<bool>
{
  QuantizeWeights(loco::DataType input, loco::DataType output, QuantizationGranularity gr)
      : input_type(input), output_type(output), granularity(gr)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;
  QuantizationGranularity granularity;

  // Quantize input tensors of each node
  bool visit(luci::CircleNode *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;
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
          // will be implemented.
        }
        // Find min/max per layer-wise
        else
        {
          // Quantize using recorded quantparam
          auto quantparam = circle_node->quantparam();
          assert(quantparam != nullptr);
          assert(quantparam->min.size() == 1);   // only support layer-wise quant
          assert(quantparam->scale.size() == 1); // only support layer-wise quant
          auto min = quantparam->min[0];
          auto scaling_factor = quantparam->scale[0];
          asym_wquant(circle_const, min, scaling_factor);
        }
      }
    }
    return false;
  }
};

} // namespace

bool QuantizeWithMinMaxPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "QuantizeWithMinMaxPass Start" << std::endl;

  // Quantize activation
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    QuantizeActivation qa(_input_dtype, _output_dtype);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qa);
  }

  // Quantize weights
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    QuantizeWeights qw(_input_dtype, _output_dtype, _granularity);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qw);
  }

  // Quantize bias
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    QuantizeBias qb(_input_dtype, _output_dtype, _granularity);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qb);
  }

  INFO(l) << "QuantizeWithMinMaxPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
