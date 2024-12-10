/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
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

#include "luci/Pass/QuantizeDequantizeWeightsWithGPTQPass.h"
#include "QuantizationUtils.h"
#include "helpers/LayerInfoMap.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>
#include <loco/IR/TensorShape.h>

#include <cmath>
#include <functional>

namespace luci
{

namespace
{

using IterFunc = std::function<void(uint32_t *, loco::TensorShape &, int32_t)>;

void iterate_per_channel_with_order(CircleConst *node, IterFunc func, bool reverse)
{
  assert(node != nullptr);

  loco::TensorShape dimension;
  dimension.rank(4);
  uint32_t indices[4] = {0};
  int32_t index_channel_dim{0};
  uint32_t num_dims[4];
  if (!get_channel_dim_index(node, dimension, index_channel_dim))
  {
    throw std::runtime_error("GPTQPass: Failed to get channel dim index.");
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
          func(indices, dimension, index_channel_dim);
        }
      }
    }
  }
}

size_t calculate_qauntized_value(CircleConst *node, uint32_t *indices, loco::TensorShape &dimension,
                                 int index_channel_dim, std::vector<float> &scaling_factor,
                                 std::vector<float> &max, std::vector<float> &min)
{
  assert(node != nullptr);

  int idx_channel = indices[index_channel_dim];

  assert(scaling_factor[idx_channel] > 0);
  const float scaling_factor_inv = 1.0 / scaling_factor[idx_channel];
  auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
  auto data_clipped = data < min[idx_channel] ? min[idx_channel] : data;
  data_clipped = data_clipped > max[idx_channel] ? max[idx_channel] : data_clipped;

  return static_cast<int32_t>(std::round((data_clipped - min[idx_channel]) * scaling_factor_inv));
}

void apply_dampening_to_hessian(std::vector<float> &hessian, uint32_t num_size)
{
  float damp = 0;
  float percdamp = .01;

  for (uint32_t i = 0; i < num_size; i++)
  {
    damp += hessian[i * num_size + i];
  }

  assert(num_size != 0);
  damp /= num_size;
  damp *= percdamp;

  for (uint32_t i = 0; i < num_size; i++)
  {
    hessian[i * num_size + i] += damp;
  }
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
          throw std::runtime_error("Error: Matrix is not positive definite.");
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
    assert(L[i * num_size + i] != 0);
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
    assert(U[i * num_size + i] != 0);
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
  int32_t index_channel_dim{0};

  if (!get_channel_dim_index(node, dimension, index_channel_dim))
  {
    throw std::runtime_error("GPTQPass: Failed to get channel dim index.");
  }
  auto size = dimension.dim(index_channel_dim).value();

  std::vector<bool> has_min_max_value(size, false);
  min.resize(size);
  max.resize(size);

  auto cal_minmax = [&](uint32_t *indices, loco::TensorShape &dimension, int index_channel_dim) {
    int idx_channel = indices[index_channel_dim];
    auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
    if (has_min_max_value[idx_channel])
    {
      min[idx_channel] = data < min[idx_channel] ? data : min[idx_channel];
      max[idx_channel] = data > max[idx_channel] ? data : max[idx_channel];
    }
    else
    {
      min[idx_channel] = data;
      max[idx_channel] = data;
      has_min_max_value[idx_channel] = true;
    }
  };

  iterate_per_channel_with_order(node, cal_minmax, false);
}

/**
 * @brief Compute the scale and zero point for the given range of values
 */
void compute_asym_scale_zp(float min, float max, loco::DataType data_type, float &scaling_factor,
                           int64_t &zp, float &nudged_min, float &nudged_max)
{
  LOGGER(l);

  assert(min <= max);
  const int32_t kMinScale = 0;
  const int32_t kMaxScale = data_type == loco::DataType::U4 ? 15 : 255;

  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const double rmin = std::fmin(0, min);
  const double rmax = std::fmax(0, max);
  const double qrange = qmax_double - qmin_double;
  assert(qrange > 0);

  double scale = (rmax - rmin) / qrange;
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
    scale = max / qrange;
    if (min > 0 && max > 0)
      WARN(l) << "The minimum and maximum values are all positive." << std::endl;
  }
  else if (max < 0)
  {
    assert(min < 0 && max < 0);
    nudged_zero_point = kMaxScale;
    scale = -min / qrange;
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

void transpose_to_upper_triangular(std::vector<float> &matrix, uint32_t num_size)
{
  for (uint32_t i = 0; i < num_size; i++)
  {
    for (uint32_t j = 0; j < i; j++)
    {
      float tmp = matrix[i * num_size + j];
      matrix[i * num_size + j] = matrix[j * num_size + i];
      matrix[j * num_size + i] = tmp;
    }
  }
}

void asymmetric_wquant_per_channel(CircleConst *node, std::vector<float> &min,
                                   std::vector<float> &max, std::vector<float> &scaling_factor,
                                   std::vector<int64_t> &zp, std::vector<float> &nudged_min,
                                   std::vector<float> &nudged_max, loco::DataType output_type)
{
  assert(node->dtype() == loco::DataType::FLOAT32);
  assert(output_type == loco::DataType::U8 || output_type == loco::DataType::U4);

  IterFunc quantize;

  const int32_t kMinScale = 0;
  const int32_t kMaxScale = output_type == loco::DataType::U4 ? 15 : 255;

  uint32_t input_size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(input_size);

  for (size_t i = 0; i < min.size(); ++i)
  {
    compute_asym_scale_zp(min[i], max[i], output_type, scaling_factor[i], zp[i], nudged_min[i],
                          nudged_max[i]);
  }

  quantize = [&](uint32_t *indices, loco::TensorShape &dimension, int index_channel_dim) {
    quantized_values[cal_offset(dimension, indices)] = calculate_qauntized_value(
      node, indices, dimension, index_channel_dim, scaling_factor, nudged_max, nudged_min);
  };
  iterate_per_channel_with_order(node, quantize, false);

  node->dtype(loco::DataType::U8);            // Change the type of tensor
  node->size<loco::DataType::U8>(input_size); // Resize tensor
  for (uint32_t i = 0; i < input_size; ++i)
  {
    node->at<loco::DataType::U8>(i) = std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void asymmetric_wquant_per_channel_with_gptq(
  CircleConst *node, std::vector<float> &min, std::vector<float> &max,
  std::vector<float> &scaling_factor, std::vector<int64_t> &zp, std::vector<float> &nudged_min,
  std::vector<float> &nudged_max, std::vector<float> &hessian, loco::DataType output_type)
{
  assert(node->dtype() == loco::DataType::FLOAT32);
  assert(output_type == loco::DataType::U8 || output_type == loco::DataType::U4);

  IterFunc quantize;

  const int32_t kMinScale = 0;
  const int32_t kMaxScale = output_type == loco::DataType::U4 ? 15 : 255;

  uint32_t input_size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(input_size);

  for (size_t i = 0; i < min.size(); ++i)
  {
    compute_asym_scale_zp(min[i], max[i], output_type, scaling_factor[i], zp[i], nudged_min[i],
                          nudged_max[i]);
  }

  uint32_t size_hessian = static_cast<uint32_t>(sqrt(hessian.size()));

  // Calculate hessian inverse
  apply_dampening_to_hessian(hessian, size_hessian);
  cholesky_decomposition(hessian, size_hessian);
  cholesky_inverse(hessian, size_hessian);
  cholesky_decomposition(hessian, size_hessian);
  transpose_to_upper_triangular(hessian, size_hessian);

  std::vector<float> error(input_size);

  loco::TensorShape dimension_channel_last;
  dimension_channel_last.rank(4);

  loco::TensorShape dimension_hessian;
  dimension_hessian.rank(4);
  dimension_hessian.dim(0).set(1);
  dimension_hessian.dim(1).set(1);
  dimension_hessian.dim(2).set(size_hessian);
  dimension_hessian.dim(3).set(size_hessian);

  quantize = [&](uint32_t *indices, loco::TensorShape &dimension_input, int index_channel_dim) {
    quantized_values[cal_offset(dimension_input, indices)] = calculate_qauntized_value(
      node, indices, dimension_input, index_channel_dim, scaling_factor, nudged_max, nudged_min);

    uint32_t indices_channel_last[4] = {
      indices[0], indices[3], indices[1], indices[2] // ohwi -> oihw
    };
    uint32_t dimension_channel_last[4] = {
      dimension_input.dim(0).value(), dimension_input.dim(3).value(),
      dimension_input.dim(1).value(), dimension_input.dim(2).value()};

    uint32_t idx_quant_column =
      dimension_channel_last[2] * dimension_channel_last[3] * indices_channel_last[1] +
      dimension_channel_last[3] * indices_channel_last[2] + indices_channel_last[3];

    uint32_t idx_channel = indices[index_channel_dim];
    uint32_t indices_diag_hessian[4] = {0, 0, idx_quant_column, idx_quant_column};

    auto idx_input_data = cal_offset(dimension_input, indices);
    auto idx_hessian = cal_offset(dimension_hessian, indices_diag_hessian);

    auto input_data = node->at<loco::DataType::FLOAT32>(idx_input_data);
    auto quantized_rvalue =
      (quantized_values[idx_input_data] - zp[idx_channel]) * scaling_factor[idx_channel];

    error[idx_input_data] = (input_data - quantized_rvalue) / hessian[idx_hessian];

    if (idx_channel == (dimension_input.dim(index_channel_dim).value() - 1))
    {
      for (uint32_t o = 0; o < dimension_channel_last[0]; o++)
      {
        for (uint32_t i = 0; i < dimension_channel_last[1]; i++)
        {
          for (uint32_t h = 0; h < dimension_channel_last[2]; h++)
          {
            for (uint32_t w = 0; w < dimension_channel_last[3]; w++)
            {
              // Convert coordination
              uint32_t indices_channel_first[4] = {o, h, w, i};
              uint32_t indices_error[4] = {o, indices[1], indices[2], indices[3]};
              uint32_t idx_ihw = dimension_channel_last[2] * dimension_channel_last[3] * i +
                                 dimension_channel_last[3] * h + w;
              uint32_t indices_hessain[4] = {0, 0, idx_quant_column, idx_ihw};

              auto _idx_h = cal_offset(dimension_hessian, indices_hessain);
              auto _idx_input_data = cal_offset(dimension_input, indices_channel_first);
              auto _idx_error = cal_offset(dimension_input, indices_error);

              // Compensate quantize error
              node->at<loco::DataType::FLOAT32>(_idx_input_data) -=
                error[_idx_error] * hessian[_idx_h];
            }
          }
        }
      }
    }
  };
  iterate_per_channel_with_order(node, quantize, true);

  node->dtype(loco::DataType::U8);            // Change the type of tensor
  node->size<loco::DataType::U8>(input_size); // Resize tensor
  for (uint32_t i = 0; i < input_size; ++i)
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

  auto dequantize = [&](uint32_t *indices, loco::TensorShape &dimension, int index_channel_dim) {
    int idx_channel = indices[index_channel_dim];
    auto data = node->at<loco::DataType::U8>(cal_offset(dimension, indices));
    dequantized_values[cal_offset(dimension, indices)] =
      static_cast<float>(data) * scaling_factor[idx_channel] + nudged_min[idx_channel];
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
 * @brief QuantizeWeightsWithGPTQ quantizes and dequantizes tensors for weights uisng GPTQ algorithm
 * @details Compensate for the quantization error and update weights using Hessian matrix
 *
 */
class QuantizeDequantizeWeightsWithGPTQ final : public luci::CircleNodeMutableVisitor<void>
{
public:
  QuantizeDequantizeWeightsWithGPTQ(
    loco::DataType input, loco::DataType output, QuantizationGranularity granularity,
    std::unordered_map<const luci::CircleNode *, std::vector<float>> *hessian_map)
    : _input_type(input), _output_type(output), _granularity(granularity), _hessian_map(hessian_map)
  {
  }

private:
  loco::DataType _input_type;
  loco::DataType _output_type;
  QuantizationGranularity _granularity;
  std::unordered_map<const luci::CircleNode *, std::vector<float>> *_hessian_map;

  void fake_quantize(luci::CircleConst *weights) const
  {
    if (_granularity != luci::QuantizationGranularity::ChannelWise)
    {
      throw std::invalid_argument("Unsupported granularity");
    }

    if (_output_type != loco::DataType::U4 && _output_type != loco::DataType::U8)
    {
      throw std::runtime_error("GPTQ quantization supports uint4/uint8");
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
                                  _output_type);
    asymmetric_wdequant_per_channel(weights, scaling_factor, nudged_min);

    auto quantparam = std::make_unique<CircleQuantParam>();
    quantparam->min = nudged_min;
    quantparam->max = nudged_max;
    quantparam->scale = scaling_factor;
    quantparam->zerop = zp;

    weights->quantparam(std::move(quantparam));
  }

  void fake_quantize_with_gptq(luci::CircleConst *weights, std::vector<float> &hessian) const
  {
    if (_granularity != luci::QuantizationGranularity::ChannelWise)
    {
      throw std::invalid_argument("Unsupported granularity");
    }

    if (_output_type != loco::DataType::U4 && _output_type != loco::DataType::U8)
    {
      throw std::runtime_error("GPTQ quantization supports uint4/uint8");
    }

    // Find min/max per channel
    std::vector<float> min;
    std::vector<float> max;

    cal_minmax_per_channel(weights, min, max);

    std::vector<float> nudged_min(min.size());
    std::vector<float> nudged_max(min.size());
    std::vector<float> scaling_factor(min.size());
    std::vector<int64_t> zp(min.size());

    asymmetric_wquant_per_channel_with_gptq(weights, min, max, scaling_factor, zp, nudged_min,
                                            nudged_max, hessian, _output_type);
    asymmetric_wdequant_per_channel(weights, scaling_factor, nudged_min);

    auto quantparam = std::make_unique<CircleQuantParam>();
    quantparam->min = nudged_min;
    quantparam->max = nudged_max;
    quantparam->scale = scaling_factor;
    quantparam->zerop = zp;

    weights->quantparam(std::move(quantparam));
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
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    auto hessian = (*_hessian_map)[node];

    fake_quantize_with_gptq(new_weights, hessian);
  }

  void visit(luci::CircleDepthwiseConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    fake_quantize(new_weights);
  }

  void visit(luci::CircleTransposeConv *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;

    if (not is_quantizable(node->filter()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);

    fake_quantize(new_weights);
  }

  void visit(luci::CircleFullyConnected *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeDequantizeWeightsWithGPTQPass visit node: " << node->name() << std::endl;
    if (not is_quantizable(node->weights()))
      return;

    auto weights = loco::must_cast<luci::CircleConst *>(node->weights());
    auto new_weights = luci::clone(weights);
    node->weights(new_weights);

    auto hessian = (*_hessian_map)[node];

    fake_quantize_with_gptq(new_weights, hessian);
  }
};

} // namespace

bool QuantizeDequantizeWeightsWithGPTQPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "QuantizeDequantizeWeightsWithGPTQ Start" << std::endl;

  if (_ctx->input_model_dtype != loco::DataType::FLOAT32)
    throw std::runtime_error("GPTQPass: Weights-only quantization supports float32 input only");

  if (_ctx->output_model_dtype != loco::DataType::U8 &&
      _ctx->output_model_dtype != loco::DataType::U4)
  {
    throw std::runtime_error("GPTQPass: GPTQ quantization supports uint4/uint8");
  }

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
    QuantizeDequantizeWeightsWithGPTQ qw(_ctx->input_model_dtype, quantize_dtype(circle_node),
                                         quantize_granularity(circle_node), _hessian_map);
    circle_node->accept(&qw);
  }

  INFO(l) << "QuantizeDequantizeWeightsWithGPTQ End" << std::endl;
  return false; // one time run
}

} // namespace luci
