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

#include "luci/Pass/QuantizeWeightsWithGPTQPass.h"
#include "QuantizationUtils.h"
#include "helpers/LayerInfoMap.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>
#include <loco/IR/TensorShape.h>

#include <iostream>
#include <cmath>
#include <functional>
#include <limits>

namespace
{

using namespace luci;
using IterFunc = std::function<void(uint32_t *, loco::TensorShape &, int32_t)>;

void iterate_per_channel_with_order(CircleConst *node, IterFunc func, bool reverse)
{
  loco::TensorShape dimension;
  dimension.rank(4);
  uint32_t indices[4] = {0};
  int32_t channel_dim_index{0};
  uint32_t num_dims[4];
  if (!get_channel_dim_index(node, dimension, channel_dim_index))
  {
    throw std::runtime_error("Failed to get channel dim index.");
  }

  auto order = reverse ? std::vector<size_t>{3, 1, 2, 0} : std::vector<size_t>{0, 1, 2, 3};

  for (uint32_t i = 0; i < 4; ++i)
  {
    num_dims[i] = dimension.dim(order[i]).value();
  }

  for (uint32_t i = 0; i < num_dims[0]; i++)
  {
    for (uint32_t j = 0; j < num_dims[1]; j++)
    {
      for (uint32_t s = 0; s < num_dims[2]; s++)
      {
        for (uint32_t t = 0; t < num_dims[3]; t++)
        {
          indices[order[0]] = i;
          indices[order[1]] = j;
          indices[order[2]] = s;
          indices[order[3]] = t;
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

size_t calculate_qauntized_value(CircleConst *node, uint32_t *indices, loco::TensorShape &dimension,
                                 int channel_dim_index, std::vector<float> &scaling_factor,
                                 std::vector<float> &max, std::vector<float> &min)
{
  int channel_idx = indices[channel_dim_index];
  const float scaling_factor_inv = 1.0 / scaling_factor[channel_idx];
  auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
  auto data_clipped = data < min[channel_idx] ? min[channel_idx] : data;
  data_clipped = data_clipped > max[channel_idx] ? max[channel_idx] : data_clipped;

  return static_cast<int32_t>(std::round((data_clipped - min[channel_idx]) * scaling_factor_inv));
}

void cholesky_decomposition(std::vector<float> &src, uint32_t num_size)
{
  for (uint32_t i = 0; i < num_size; i++)
  {
    for (uint32_t j = 0; j <= i; j++)
    {
      double sum = 0;
      for (uint32_t k = 0; k < j; k++)
      {
        sum += src[i * num_size + k] * src[j * num_size + k];
      }
      if (i == j)
      {
        if (src[i * num_size + i] - sum <= 0)
        {
          std::cout << "Error: Matrix is not positive definite.\n" << std::endl;
          return;
        }
        src[i * num_size + i] = sqrt(src[i * num_size + i] - sum);
      }
      else
      {
        src[i * num_size + j] = (src[i * num_size + j] - sum) / src[j * num_size + j];
      }
    }
  }
  for (uint32_t i = 0; i < num_size; i++)
  {
    for (uint32_t j = 0; j < num_size; j++)
    {
      if (i < j)
      {
        src[i * num_size + j] = 0.0;
      }
    }
  }
  return;
}

void forward_substitution(const std::vector<float> &L, const std::vector<float> &b,
                          std::vector<float> &y, int num_size)
{
  for (int i = 0; i < num_size; ++i)
  {
    y[i] = b[i];
    for (int j = 0; j < i; ++j)
    {
      y[i] -= L[i * num_size + j] * y[j];
    }
    y[i] /= L[i * num_size + i];
  }
}

void backward_substitution(const std::vector<float> &U, const std::vector<float> &y,
                           std::vector<float> &x, int num_size)
{
  for (int i = num_size - 1; i >= 0; --i)
  {
    x[i] = y[i];
    for (int j = i + 1; j < num_size; ++j)
    {
      x[i] -= U[i * num_size + j] * x[j];
    }
    x[i] /= U[i * num_size + i];
  }
}

void cholesky_inverse(std::vector<float> &L, uint32_t num_size)
{
  std::vector<float> L_inv(L.size());
  std::vector<float> H_inv(L.size());

  std::vector<float> e(num_size, 0);
  std::vector<float> col(num_size, 0);
  std::vector<float> temp(num_size, 0);

  for (uint32_t i = 0; i < num_size; ++i)
  {
    fill(e.begin(), e.end(), 0.0);
    e[i] = 1.0;

    forward_substitution(L, e, temp, num_size);

    for (uint32_t j = 0; j < num_size; ++j)
    {
      L_inv[j * num_size + i] = temp[j];
    }
  }

  for (uint32_t i = 0; i < num_size; i++)
  {
    for (uint32_t j = 0; j < i; j++)
    {
      float tmp = L[i * num_size + j];
      L[i * num_size + j] = L[j * num_size + i];
      L[j * num_size + i] = tmp;
    }
  }

  for (uint32_t i = 0; i < num_size; ++i)
  {
    fill(e.begin(), e.end(), 0.0);
    fill(col.begin(), col.end(), 0.0);
    e[i] = 1.0;
    for (uint32_t j = 0; j < num_size; j++)
    {
      col[j] = L_inv[j * num_size + i];
    }
    backward_substitution(L, col, temp, num_size);
    for (uint32_t j = 0; j < num_size; ++j)
    {
      H_inv[j * num_size + i] = temp[j];
    }
  }
  for (uint32_t i = 0; i < L.size(); i++)
  {
    L[i] = H_inv[i];
  }
}

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

  iterate_per_channel_with_order(node, cal_minmax, false);
}

void compute_asym_scale_zp(float min, float max, float &scaling_factor, int64_t &zp,
                           float &nudged_min, float &nudged_max, int32_t k_max_scale)
{
  LOGGER(l);

  assert(min <= max);
  const int32_t kMinScale = 0;
  const int32_t kMaxScale = k_max_scale;
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

void asymmetric_wquant_per_channel(CircleConst *node, std::vector<float> &min,
                                   std::vector<float> &max, std::vector<float> &scaling_factor,
                                   std::vector<int64_t> &zp, std::vector<float> &nudged_min,
                                   std::vector<float> &nudged_max, loco::DataType output_type,
                                   std::vector<float> &hessian)
{
  assert(node->dtype() == loco::DataType::FLOAT32);

  IterFunc quantize;

  const int32_t kMinScale = 0;
  const int32_t kMaxScale = output_type == loco::DataType::U4 ? 15 : 255;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(size);

  for (size_t i = 0; i < min.size(); ++i)
  {
    compute_asym_scale_zp(min[i], max[i], scaling_factor[i], zp[i], nudged_min[i], nudged_max[i],
                          kMaxScale);
  }

  if (hessian.empty()) // Cases where gptq is not applied
  {
    quantize = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
      quantized_values[cal_offset(dimension, indices)] = calculate_qauntized_value(
        node, indices, dimension, channel_dim_index, scaling_factor, nudged_max, nudged_min);
    };
    iterate_per_channel_with_order(node, quantize, false);
  }
  else // Cases where gptq is applied
  {
    uint32_t size_hessian = static_cast<uint32_t>(sqrt(hessian.size()));
    float percdamp = .01;
    float damp = 0;

    for (uint32_t i = 0; i < size_hessian; i++)
    {
      damp += hessian[i * size_hessian + i];
    }
    damp /= size_hessian;
    damp *= percdamp;

    for (uint32_t i = 0; i < size_hessian; i++)
    {
      hessian[i * size_hessian + i] += damp;
    }

    // calculate hessian inverse
    cholesky_decomposition(hessian, size_hessian);
    cholesky_inverse(hessian, size_hessian);
    cholesky_decomposition(hessian, size_hessian);

    // transpose hessian to make upper trangular
    for (uint32_t i = 0; i < size_hessian; i++)
    {
      for (uint32_t j = 0; j < i; j++)
      {
        float tmp = hessian[i * size_hessian + j];
        hessian[i * size_hessian + j] = hessian[j * size_hessian + i];
        hessian[j * size_hessian + i] = tmp;
      }
    }

    std::vector<float> error(size);

    loco::TensorShape dimension_channel_last;
    dimension_channel_last.rank(4);

    loco::TensorShape dimension_hessian;
    dimension_hessian.rank(2);
    dimension_hessian.dim(0).set(size_hessian);
    dimension_hessian.dim(1).set(size_hessian);

    quantize = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
      quantized_values[cal_offset(dimension, indices)] = calculate_qauntized_value(
        node, indices, dimension, channel_dim_index, scaling_factor, nudged_max, nudged_min);

      uint32_t indices_channel_last[4] = {
        indices[0], indices[3], indices[1], indices[2] // ohwi -> oihw
      };

      uint32_t dimension_channel_last[4] = {dimension.dim(0).value(), dimension.dim(3).value(),
                                            dimension.dim(1).value(), dimension.dim(2).value()};

      uint32_t idx_quant_column =
        dimension_channel_last[2] * dimension_channel_last[3] * indices_channel_last[1] +
        dimension_channel_last[3] * indices_channel_last[2] + indices_channel_last[3];

      uint32_t indices_diag_hessian[2] = {idx_quant_column, idx_quant_column};

      uint32_t channel_idx = indices[channel_dim_index];
      auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));

      error[cal_offset(dimension, indices)] =
        (data - (quantized_values[cal_offset(dimension, indices)] - zp[channel_idx]) *
                  scaling_factor[channel_idx]) /
        hessian[cal_offset_2d(dimension_hessian, indices_diag_hessian)];

      if (channel_idx == (dimension.dim(channel_dim_index).value() - 1))
      {
        for (uint32_t o = 0; o < dimension_channel_last[0]; o++)
        {
          for (uint32_t i = 0; i < dimension_channel_last[1]; i++)
          {
            for (uint32_t h = 0; h < dimension_channel_last[2]; h++)
            {
              for (uint32_t w = 0; w < dimension_channel_last[3]; w++)
              {
                // convert coordination
                uint32_t indices_channel_first[4] = {o, h, w, i};
                uint32_t indices_error[4] = {o, indices[1], indices[2], indices[3]};
                uint32_t idx_ihw = dimension_channel_last[2] * dimension_channel_last[3] * i +
                                   dimension_channel_last[3] * h + w;
                uint32_t indices_hessain[2] = {idx_quant_column, idx_ihw};

                node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices_channel_first)) -=
                  error[cal_offset(dimension, indices_error)] *
                  hessian[cal_offset_2d(dimension_hessian, indices_hessain)];
              }
            }
          }
        }
      }
    };
    iterate_per_channel_with_order(node, quantize, true);
  }

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

  iterate_per_channel_with_order(node, dequantize, false);

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
struct QuantizeWeightsWithGPTQ final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeWeightsWithGPTQ(
    loco::DataType input, loco::DataType output, QuantizationGranularity granularity,
    std::unordered_map<const luci::CircleNode *, std::vector<float>> *hessian_map)
    : input_type(input), output_type(output), granularity(granularity), hessian_map(hessian_map)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;
  QuantizationGranularity granularity;
  std::unordered_map<const luci::CircleNode *, std::vector<float>> *hessian_map;

private:
  void fake_quantize_cwq(luci::CircleConst *weights, std::vector<float> &hessian) const
  {
    // assert(output_type == loco::DataType::U8); // FIX_CALLER_UNLESS
    if (output_type != loco::DataType::U8)
    {
        throw std::runtime_error("GPTQ quantization supports u8");
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
                                  output_type, hessian);
    asymmetric_wdequant_per_channel(weights, scaling_factor, nudged_min);

    auto quantparam = std::make_unique<CircleQuantParam>();
    quantparam->min = nudged_min;
    quantparam->max = nudged_max;
    quantparam->scale = scaling_factor;
    quantparam->zerop = zp;

    weights->quantparam(std::move(quantparam));
  }

  void fake_quantize(luci::CircleConst *weights, std::vector<float> &hessian) const
  {
    switch (granularity)
    {
      case luci::QuantizationGranularity::ChannelWise:
        fake_quantize_cwq(weights, hessian);
        break;
      default:
        throw std::invalid_argument("Unsupported granularity");
    }
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
    INFO(l) << "QuantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    auto hessian = (*hessian_map)[node];

    fake_quantize(new_weights, hessian);
  }

  void visit(luci::CircleDepthwiseConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    std::vector<float> empty_vector;

    fake_quantize(new_weights, empty_vector);
  }

  void visit(luci::CircleTransposeConv *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeights visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    std::vector<float> empty_vector;

    fake_quantize(new_weights, empty_vector);
  }

  void visit(luci::CircleFullyConnected *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeights visit node: " << node->name() << std::endl;
    if (not is_quantizable(node->weights()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->weights());
    auto new_weights = luci::clone(weights);
    node->weights(new_weights);

    auto hessian = (*hessian_map)[node];

    fake_quantize(new_weights, hessian);
  }
};

} // namespace

bool QuantizeWeightsWithGPTQPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeightsWithGPTQPass Start" << std::endl;

  if (_ctx->input_model_dtype != loco::DataType::FLOAT32)
    throw std::runtime_error("Weights-only quantization supports float32 input only");

  if (_ctx->output_model_dtype != loco::DataType::U8)
    throw std::runtime_error("GPTQ quantization supports uint8 output only");

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
    QuantizeWeightsWithGPTQ qw(_ctx->input_model_dtype, quantize_dtype(circle_node),
                               quantize_granularity(circle_node), _hessian_map);
    circle_node->accept(&qw);
  }

  INFO(l) << "QuantizeWeightsWithGPTQPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
