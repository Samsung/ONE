/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "QuantizationUtils.h"

#include <luci/Log.h>

#include <iostream>
#include <cmath>

namespace luci
{

bool is_quantized(const CircleNode *node)
{
  return node->quantparam() != nullptr &&
         (node->dtype() == loco::DataType::U8 ||  // activation, weight (uint8 quant)
          node->dtype() == loco::DataType::S16 || // activation, weight (int16 quant)
          node->dtype() == loco::DataType::S32 || // bias (uint8 quant)
          node->dtype() == loco::DataType::S64);  // bias (int16 quant)
}

// Check if node is weights of conv2d, depthwise_conv2d, or fully_connected layer
bool is_weights(CircleNode *node)
{
  auto circle_const = dynamic_cast<CircleConst *>(node);
  if (circle_const == nullptr)
    return false;

  auto succs = loco::succs(node);

  // Node is weights if it is the weights of all of its successors
  for (auto out : succs)
  {
    bool is_weights = false;

    auto conv = dynamic_cast<CircleConv2D *>(out);
    if (conv != nullptr && conv->filter() == circle_const)
      is_weights = true;

    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    if (dw_conv != nullptr && dw_conv->filter() == circle_const)
      is_weights = true;

    auto t_conv = dynamic_cast<CircleTransposeConv *>(out);
    if (t_conv != nullptr && t_conv->filter() == circle_const && circle_const->rank() == 4)
      is_weights = true;

    auto fc = dynamic_cast<CircleFullyConnected *>(out);
    if (fc != nullptr && fc->weights() == circle_const)
      is_weights = true;

    if (!is_weights)
      return false;
  }

  return true;
}

uint8_t fp32_to_uint8_cast(float f)
{
  assert(std::numeric_limits<uint8_t>::min() <= f);
  assert(f <= std::numeric_limits<uint8_t>::max());
  return static_cast<uint8_t>(f);
}

// Per-layer quantization of weights (const tensor) using given min/max values
void asymmetric_wquant_with_minmax_per_layer(CircleConst *node, float min, float max,
                                             float &scaling_factor, int64_t &zp, float &nudged_min,
                                             float &nudged_max)
{
  const int32_t kMinScale = 0;
  const int32_t kMaxScale = 255;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
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

// Per-layer quantization of weights (const tensor) using given min/max values
void symmetric_wquant_with_minmax_per_layer(CircleConst *node, float min, float max,
                                            float &scaling_factor, int64_t &zp, float &nudged_min,
                                            float &nudged_max)
{
  const int32_t kMaxScale = std::numeric_limits<int16_t>::max();
  const int32_t kMinScale = -kMaxScale;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  compute_sym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
  const float scaling_factor_inv = 1.0 / scaling_factor;
  std::vector<int32_t> quantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    // clipping
    auto data = node->at<loco::DataType::FLOAT32>(i);
    data = data < nudged_min ? nudged_min : data;
    data = data > nudged_max ? nudged_max : data;
    quantized_values[i] = static_cast<int32_t>(std::round(data * scaling_factor_inv));
  }

  node->dtype(loco::DataType::S16);      // change the type of tensor
  node->size<loco::DataType::S16>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::S16>(i) =
        std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void compute_sym_scale_zp(float min, float max, float &scaling_factor, int64_t &zp,
                          float &nudged_min, float &nudged_max)
{
  assert(min != max);

  const int32_t kMaxScale = std::numeric_limits<int16_t>::max();
  const int32_t kMinScale = -kMaxScale;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const double rmin = std::fmin(0, min);
  const double rmax = std::fmax(0, max);
  double scale_factor_from_min_side{0};
  double scale_factor_from_max_side{0};

  if ((qmin_double * rmin) > 0)
    scale_factor_from_min_side = rmin / qmin_double;

  if ((qmax_double * rmax) > 0)
    scale_factor_from_max_side = rmax / qmax_double;

  scaling_factor = scale_factor_from_min_side > scale_factor_from_max_side
                       ? scale_factor_from_min_side
                       : scale_factor_from_max_side;
  zp = 0;
  nudged_min = static_cast<float>(qmin_double * scaling_factor);
  nudged_max = static_cast<float>(qmax_double * scaling_factor);
}

void compute_asym_scale_zp(float min, float max, float &scaling_factor, int64_t &zp,
                           float &nudged_min, float &nudged_max)
{
  LOGGER(l);

  assert(min <= max);
  const int32_t kMinScale = 0;
  const int32_t kMaxScale = 255;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const double rmin = std::fmin(0, min);
  const double rmax = std::fmax(0, max);

  double scale = (rmax - rmin) / (qmax_double - qmin_double);
  double zero_point_double = 0;
  uint8_t nudged_zero_point = 0;
  if (scale == 0)
  {
    WARN(l) << "The minimum and maximum values are the same." << std::endl;
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
    scale = max / (qmax_double - qmin_double);
    if (min > 0 && max > 0)
      WARN(l) << "The minimum and maximum values are all positive." << std::endl;
  }
  else if (max < 0)
  {
    assert(min < 0 && max < 0);
    nudged_zero_point = kMaxScale;
    scale = -min / (qmax_double - qmin_double);
    WARN(l) << "The minimum and maximum values are all negative." << std::endl;
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

bool get_channel_dim_index(CircleConst *node, loco::TensorShape &dimension, int &channel_dim_index)
{
  auto succs = loco::succs(node);

  // opcode is initialized to CIRCLEINPUT, because
  // CIRCLEINPUT should never be the successor of any node
  // (this is checked w/ the assert in the loop body)
  luci::CircleOpcode opcode = luci::CircleOpcode::CIRCLEINPUT;
  for (auto out : succs)
  {
    const auto circle_node = static_cast<CircleNode *>(out);
    assert(circle_node->opcode() != luci::CircleOpcode::CIRCLEINPUT);

    if (opcode == luci::CircleOpcode::CIRCLEINPUT)
    {
      opcode = circle_node->opcode();
    }
    else
    {
      // Node is used by multiple layers with different opcodes
      // We do not care such cases
      if (opcode != circle_node->opcode())
        return false;
    }
  }

  for (auto out : succs)
  {
    auto conv = dynamic_cast<CircleConv2D *>(out);
    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    auto tw_conv = dynamic_cast<CircleTransposeConv *>(out);
    auto fc = dynamic_cast<CircleFullyConnected *>(out);

    // Refer to https://github.com/Samsung/ONE/pull/2448.
    if ((conv != nullptr && conv->filter() == node) ||
        (tw_conv != nullptr && tw_conv->filter() == node)) // OHWI
    {
      assert(node->rank() == 4);
      dimension.dim(0).set(node->dim(0).value());
      dimension.dim(1).set(node->dim(1).value());
      dimension.dim(2).set(node->dim(2).value());
      dimension.dim(3).set(node->dim(3).value());
      channel_dim_index = 0; // Set channel_dim_index based on "O"
      return true;
    }
    else if (dw_conv != nullptr && dw_conv->filter() == node) // IHWC
    {
      assert(node->rank() == 4);
      dimension.dim(0).set(node->dim(0).value());
      dimension.dim(1).set(node->dim(1).value());
      dimension.dim(2).set(node->dim(2).value());
      dimension.dim(3).set(node->dim(3).value());
      channel_dim_index = 3; // Set channel_dim_index based on "C"
      return true;
    }
    else if (fc != nullptr && fc->weights() == node) // OI
    {
      assert(node->rank() == 2);
      dimension.dim(0).set(node->dim(0).value());
      dimension.dim(1).set(1); // Set FC layer like CONV
      dimension.dim(2).set(1);
      dimension.dim(3).set(node->dim(1).value());
      channel_dim_index = 0; // Set channel_dim_index based on "O"
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

uint32_t cal_offset(loco::TensorShape &dimension, uint32_t *indices) // TODO : value() fix needed
{
  return indices[0] * dimension.dim(1).value() * dimension.dim(2).value() *
             dimension.dim(3).value() +
         indices[1] * dimension.dim(2).value() * dimension.dim(3).value() +
         indices[2] * dimension.dim(3).value() + indices[3];
}

} // namespace luci
