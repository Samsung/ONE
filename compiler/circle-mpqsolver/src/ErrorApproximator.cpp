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

#include "ErrorApproximator.h"

#include <cmath>
#include <limits>
#include <vector>
#include <functional>
#include <luci/IR/Nodes/CircleConst.h>
#include <luci/IR/Nodes/CircleConv2D.h>
#include <luci/IR/Nodes/CircleTransposeConv.h>
#include <luci/IR/Nodes/CircleFullyConnected.h>
#include <luci/IR/Nodes/CircleDepthwiseConv2D.h>

using namespace mpqsolver;

namespace
{

using namespace luci;
using IterFunc = std::function<void(uint32_t *, loco::TensorShape &, int32_t)>;

inline bool has_min_max(const CircleNode *node)
{
  return node->quantparam() && !node->quantparam()->min.empty() && !node->quantparam()->max.empty();
}

inline uint32_t cal_offset(const loco::TensorShape &dimension, uint32_t *indices)
{
  return indices[0] * dimension.dim(1).value() * dimension.dim(2).value() *
           dimension.dim(3).value() +
         indices[1] * dimension.dim(2).value() * dimension.dim(3).value() +
         indices[2] * dimension.dim(3).value() + indices[3];
}

uint32_t get_channel_dim_index(const CircleNode *node)
{
  uint32_t index = 0;
  auto opcode = node->opcode();
  switch (opcode)
  {
    case CircleOpcode::CONV_2D:
    case CircleOpcode::TRANSPOSE_CONV:
    case CircleOpcode::FULLY_CONNECTED:
    {
      index = 0;
    }
    break;
    case CircleOpcode::DEPTHWISE_CONV_2D:
    {
      index = 3;
    }
    break;
    default:
      throw std::runtime_error("Failed to find channel index in " + node->name());
  }

  return index;
}

bool set_weight_dim(const CircleNode *node, const CircleConst *weights,
                    loco::TensorShape &dimension)
{
  auto opcode = node->opcode();
  switch (opcode)
  {
    case CircleOpcode::CONV_2D:
    case CircleOpcode::TRANSPOSE_CONV:
    case CircleOpcode::DEPTHWISE_CONV_2D:
    {
      assert(node->rank() == 4);
      dimension.rank(node->rank());
      dimension.dim(0).set(weights->dim(0).value());
      dimension.dim(1).set(weights->dim(1).value());
      dimension.dim(2).set(weights->dim(2).value());
      dimension.dim(3).set(weights->dim(3).value());
    }
    break;
    case CircleOpcode::FULLY_CONNECTED:
    {
      assert(node->rank() == 2);
      dimension.rank(4);
      dimension.dim(0).set(weights->dim(0).value());
      dimension.dim(1).set(1); // Set FC layer like CONV
      dimension.dim(2).set(1);
      dimension.dim(3).set(weights->dim(1).value());
    }
    break;
    default:
      return false;
  }

  return true;
}

loco::Node *get_weight(const CircleNode *node)
{
  loco::Node *weight = nullptr;
  auto opcode = node->opcode();
  switch (opcode)
  {
    case CircleOpcode::CONV_2D:
    {
      auto conv = loco::must_cast<const CircleConv2D *>(node);
      weight = conv->filter();
    }
    break;
    case CircleOpcode::DEPTHWISE_CONV_2D:
    {
      auto dconv = loco::must_cast<const CircleDepthwiseConv2D *>(node);
      weight = dconv->filter();
    }
    break;
    case CircleOpcode::TRANSPOSE_CONV:
    {
      auto tconv = loco::must_cast<const CircleTransposeConv *>(node);
      weight = tconv->filter();
    }
    break;
    case CircleOpcode::FULLY_CONNECTED:
    {
      auto fc = loco::must_cast<const CircleFullyConnected *>(node);
      weight = fc->weights();
    }
    break;
    default:
      break;
  }

  return weight;
}

void iterate_per_channel(const CircleNode *node, IterFunc func)
{
  CircleConst *weight = loco::must_cast<CircleConst *>(get_weight(node));
  loco::TensorShape dimension;
  set_weight_dim(node, weight, dimension);
  uint32_t indices[4] = {
    0,
  };

  auto channel_dim_index = get_channel_dim_index(node);

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

void cal_minmax_per_channel(const CircleNode *node, std::vector<float> &min,
                            std::vector<float> &max)
{
  CircleConst *weight = loco::must_cast<CircleConst *>(get_weight(node));
  loco::TensorShape dimension;
  set_weight_dim(node, weight, dimension);

  auto channel_dim_index = get_channel_dim_index(node);
  auto size = dimension.dim(channel_dim_index).value();

  std::vector<bool> has_min_max_value(size, false);
  min.resize(size);
  max.resize(size);

  auto cal_minmax = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    auto data = weight->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
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

bool get_shape(const CircleNode *circle_node, std::vector<uint32_t> &shape)
{
  if (circle_node->shape_status() == ShapeStatus::VALID)
  {
    auto rank = circle_node->rank();
    if (rank != 4)
      return false;

    shape.resize(rank);
    for (int i = 0; i < rank; i++)
    {
      shape[i] = circle_node->dim(i).value();
    }
    return true;
  }

  return false;
}

const CircleNode *get_activation_node(const CircleNode *node)
{
  // only convolutions are considered currently, so just one input
  auto preds = loco::preds(node);
  for (const auto &pred : preds)
  {
    auto cur_node = loco::must_cast<const CircleNode *>(pred);
    if (cur_node->opcode() != CircleOpcode::CIRCLECONST)
    {
      return cur_node;
    }
  }
  return nullptr;
}

uint32_t get_stride_multiplier(const CircleNode *node)
{
  uint32_t mult = 1;
  if (node->opcode() == CircleOpcode::DEPTHWISE_CONV_2D)
  {
    auto dconv = loco::must_cast<const CircleDepthwiseConv2D *>(node);
    auto stride = dconv->stride();
    mult *= (stride->h() * stride->w());
  }
  else if (node->opcode() == CircleOpcode::CONV_2D)
  {
    auto nconv = loco::must_cast<const CircleConv2D *>(node);
    auto stride = nconv->stride();
    mult *= (stride->h() * stride->w());
  }
  else if (node->opcode() == CircleOpcode::TRANSPOSE_CONV)
  {
    auto nconv = loco::must_cast<const CircleTransposeConv *>(node);
    auto stride = nconv->stride();
    mult *= (stride->h() * stride->w());
  }

  return mult;
}

uint32_t get_computations_per_channel(const CircleNode *node)
{
  uint32_t computations_per_channel = 1;
  std::vector<uint32_t> cir_shape;
  // size of activation map
  auto activation = get_activation_node(node);
  assert(activation != nullptr);
  if (get_shape(activation, cir_shape))
  {
    for (uint32_t index = 0; index < cir_shape.size(); index++)
    {
      computations_per_channel *= cir_shape[index];
    }
  }

  if (auto weights = loco::must_cast<CircleNode *>(get_weight(node)))
  {
    std::vector<uint32_t> w_shape;
    if (get_shape(weights, w_shape))
    {
      computations_per_channel *= (w_shape[1] * w_shape[2]); //[channels_out, k_x, k_y, channels_in]
    }
  }

  uint32_t stride_mult = get_stride_multiplier(node);

  return computations_per_channel / stride_mult;
}

void get_min_max_activation_values(const CircleNode *node, float &ci_min, float &ci_max)
{
  auto preds = loco::preds(node);
  for (const auto &pred : preds)
  {
    auto parent_node = loco::must_cast<const luci::CircleNode *>(pred);
    if (has_min_max(parent_node))
    {
      auto quantparam = parent_node->quantparam();
      if (quantparam->min.size() > 0)
      {
        ci_min = quantparam->min[0];
        ci_max = quantparam->max[0];
      }
    }
  }
}

} // namespace

/**
 * How Approximate works?
 *
 * Currently it works just for convolution layers, but may be generalized for other types as well.
 * See discussion at https://github.com/Samsung/ONE/pull/10170#discussion_r1042246598
 * Convolution is just a matrix multiplication.
 * While quantizing we introduce quantization error into convolution operand (activations) as well
 * as into convolution weights. A_q * W_q = (A + q_err(A)) * (W + q_err(W)) = A * W + A * q_err(W) +
 * W * q_err(A), q_err - quantization error, W - weight matrix, A - activations from previous layer,
 * so quantization error of matrix multiplication is A * q_err(W) + W * q_err(A).
 * Estimating its largest value we get
 * ~ number_of_additions * (A_max * (W_max - W_min) / 255 + W_max * (A_max - A_min) / 255)
 * The following code tries to get total error for quantizing convolution node into Q8.
 * It's just an heuristic (Metric sensitivity depends highly on derivatives as well).
 */
float ErrorApproximator::approximate(const CircleNode *node)
{
  auto opcode = node->opcode();
  if (opcode != CircleOpcode::CONV_2D && opcode != CircleOpcode::TRANSPOSE_CONV &&
      opcode != CircleOpcode::DEPTHWISE_CONV_2D)
  {
    return 0.0; // TODO
  }

  float volume_W_A_err = 0.f;
  {
    // activation min-max values
    float ci_min = 0.f;
    float ci_max = 0.f;
    get_min_max_activation_values(node, ci_min, ci_max);

    // channel-wise min, max
    std::vector<float> min_values;
    std::vector<float> max_values;
    cal_minmax_per_channel(node, min_values, max_values);

    // ranges  = (max_values - min_values)
    std::vector<float> ranges = max_values;
    for (uint32_t i = 0; i < min_values.size(); ++i)
    {
      ranges[i] -= min_values[i];
    }

    // maximal weight value across all channels
    float w_max = 0;
    {
      for (int i = 0; i < (int)max_values.size(); ++i)
      {
        w_max = std::max(w_max, (float)::fabs(max_values[i]));
        w_max = std::max(w_max, (float)::fabs(min_values[i]));
      }
    }

    // total weight quantization error across all channels
    // so maximal error of quantization is ~ (max_value - min_value) / 255
    // omitting 255 term we get that maximal error of quantization is just its range
    float sum_err = 0.f;
    for (auto cur_err : ranges)
    {
      sum_err += cur_err;
    }

    uint32_t computations_per_channel = get_computations_per_channel(node);
    uint32_t num_of_channels = ranges.size();

    // maximal error introduced by weights quantization (for all channels)
    volume_W_A_err = sum_err * std::max(::fabs(ci_max), ::fabs(ci_min));
    // plus total error introduced by activation quantization (for all channels)
    volume_W_A_err += w_max * num_of_channels * ::fabs(ci_max - ci_min);
    // scale by volume of computations per channel
    volume_W_A_err *= computations_per_channel;
    // scale to get more readable output values
    volume_W_A_err /= 1.e+6f;
  }

  return volume_W_A_err;
}
