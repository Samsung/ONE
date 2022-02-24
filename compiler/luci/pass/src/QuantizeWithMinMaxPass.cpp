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
#include "luci/Pass/PropagateQParamForwardPass.h"
#include "luci/Pass/PropagateQParamBackwardPass.h"
#include "QuantizationUtils.h"
#include "ProgressReporter.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Log.h>
#include <logo/Phase.h>

#include <oops/UserExn.h>

#include <iostream>
#include <cmath>
#include <functional>

namespace
{

using namespace luci;
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

// Create a Quantize Op whose
// dtype is out_type
// shape is the same with node
// qparam is computed using node's min/max
luci::CircleQuantize *create_quantize_op(luci::CircleNode *node, loco::DataType out_type)
{
  auto quantize = node->graph()->nodes()->create<CircleQuantize>();
  quantize->name(node->name() + "_Quantize");
  quantize->dtype(out_type);
  quantize->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); i++)
    quantize->dim(i).set(node->dim(i).value());

  quantize->shape_status(luci::ShapeStatus::VALID);

  auto qparam = node->quantparam();
  assert(qparam);                  // FIX_CALLER_UNLESS
  assert(qparam->min.size() == 1); // FIX_CALLER_UNLESS
  assert(qparam->max.size() == 1); // FIX_CALLER_UNLESS
  auto min = qparam->min[0];
  auto max = qparam->max[0];

  float scaling_factor{0};
  int64_t zp{0};
  float nudged_min{0};
  float nudged_max{0};

  if (out_type == loco::DataType::U8)
  {
    compute_asym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
  }
  else
  {
    assert(out_type == loco::DataType::S16);
    compute_sym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
  }

  auto quantparam = std::make_unique<CircleQuantParam>();
  quantparam->scale.push_back(scaling_factor);
  quantparam->zerop.push_back(zp);

  quantize->quantparam(std::move(quantparam));

  return quantize;
}

} // namespace

namespace luci
{

namespace
{

// Create a new const node from an existing node.
// The new node has the following characteristics
// type: T
// shape: same with 'node' (given as an argument)
// buffer size: 'size' (given as an argument)
// Note that contents are not filled in this function.
template <loco::DataType T>
luci::CircleConst *create_empty_const_from(luci::CircleConst *node, uint32_t size)
{
  auto new_node = node->graph()->nodes()->create<CircleConst>();
  // TODO: We don't have any naming convention for quantized nodes yet.
  //       Fix this when we have one.
  new_node->name(node->name());
  new_node->dtype(T);
  new_node->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); i++)
    new_node->dim(i).set(node->dim(i).value());

  new_node->size<T>(size);
  new_node->shape_status(luci::ShapeStatus::VALID);

  return new_node;
}

void overwrite_quantparam(luci::CircleNode *source, luci::CircleNode *target)
{
  auto source_qparam = source->quantparam();
  if (source_qparam == nullptr)
    throw std::runtime_error("source quantparam is not found during overwrite");

  auto target_qparam = target->quantparam();
  if (target_qparam == nullptr)
  {
    auto quantparam = std::make_unique<CircleQuantParam>();
    target->quantparam(std::move(quantparam));
    target_qparam = target->quantparam();

    if (target_qparam == nullptr)
      throw std::runtime_error("Creating new quant param failed");
  }
  target_qparam->min = source_qparam->min;
  target_qparam->max = source_qparam->max;
  target_qparam->scale = source_qparam->scale;
  target_qparam->zerop = source_qparam->zerop;
  target_qparam->quantized_dimension = source_qparam->quantized_dimension;
}

void quant_const_values(luci::CircleConst *const_node, float scaling_factor, float zerop,
                        loco::DataType quant_type)
{
  uint32_t size = const_node->size<loco::DataType::FLOAT32>();

  const float scaling_factor_inv = 1.0 / scaling_factor;
  std::vector<int32_t> quantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    auto data = static_cast<double>(const_node->at<loco::DataType::FLOAT32>(i));
    double quantized_float = std::round(data * scaling_factor_inv) + zerop;
    constexpr auto int_max = static_cast<double>(std::numeric_limits<int32_t>::max());
    constexpr auto int_min = static_cast<double>(std::numeric_limits<int32_t>::min());
    quantized_float = std::min(int_max, std::max(int_min, quantized_float));

    quantized_values[i] = static_cast<int32_t>(quantized_float);
  }

  switch (quant_type)
  {
    case loco::DataType::U8:
      const_node->dtype(loco::DataType::U8);      // change the type of tensor
      const_node->size<loco::DataType::U8>(size); // resize tensor
      for (uint32_t i = 0; i < size; ++i)
        const_node->at<loco::DataType::U8>(i) = std::min(255, std::max(0, quantized_values[i]));
      break;
    case loco::DataType::S16:
      assert(zerop == 0);
      const_node->dtype(loco::DataType::S16);      // change the type of tensor
      const_node->size<loco::DataType::S16>(size); // resize tensor
      for (uint32_t i = 0; i < size; ++i)
        const_node->at<loco::DataType::S16>(i) =
          std::min(32767, std::max(-32767, quantized_values[i]));
      break;
    default:
      throw std::runtime_error("Unsupported data type");
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

void quant_const(CircleConst *node, loco::DataType quant_type)
{
  assert(node->dtype() == loco::DataType::FLOAT32);

  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  for (uint32_t i = 0; i < node->size<loco::DataType::FLOAT32>(); i++)
  {
    auto data = node->at<loco::DataType::FLOAT32>(i);
    min = data < min ? data : min;
    max = data > max ? data : max;
  }

  float scaling_factor{0.0};
  int64_t zp{0};
  float nudged_min{0.0};
  float nudged_max{0.0};

  switch (quant_type)
  {
    case loco::DataType::U8:
      asymmetric_wquant_with_minmax_per_layer(node, min, max, scaling_factor, zp, nudged_min,
                                              nudged_max);
      break;
    case loco::DataType::S16:
      symmetric_wquant_with_minmax_per_layer(node, min, max, scaling_factor, zp, nudged_min,
                                             nudged_max);
      break;
    default:
      throw std::runtime_error("Unsupported data type");
  }

  auto quantparam = std::make_unique<CircleQuantParam>();
  quantparam->scale.push_back(scaling_factor);
  quantparam->zerop.push_back(zp);
  node->quantparam(std::move(quantparam));
}

// Check if the node is the bias of Conv2D, DepthwiseConv2D, FullyConnected, or TransposeConv layer
// Returns a list of <input, weights, output> vectors for the above operators.
// Note that it returns a 'list' because bias can be used by multiple operators.
std::vector<std::vector<loco::Node *>> get_input_weight_output_of_bias(CircleNode *node)
{
  std::vector<std::vector<loco::Node *>> result;
  auto circle_const = dynamic_cast<CircleConst *>(node);
  if (circle_const == nullptr)
    return result;

  auto succs = loco::succs(node);

  for (auto out : succs)
  {
    auto conv = dynamic_cast<CircleConv2D *>(out);
    if (conv != nullptr && conv->bias() == circle_const)
    {
      assert(conv->input() != nullptr);
      assert(conv->filter() != nullptr);
      result.push_back({conv->input(), conv->filter(), conv});
      continue;
    }
    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    if (dw_conv != nullptr && dw_conv->bias() == circle_const)
    {
      assert(dw_conv->input() != nullptr);
      assert(dw_conv->filter() != nullptr);
      result.push_back({dw_conv->input(), dw_conv->filter(), dw_conv});
      continue;
    }
    auto fc = dynamic_cast<CircleFullyConnected *>(out);
    if (fc != nullptr && fc->bias() == circle_const)
    {
      assert(fc->input() != nullptr);
      assert(fc->weights() != nullptr);
      result.push_back({fc->input(), fc->weights(), fc});
      continue;
    }
    auto tconv = dynamic_cast<CircleTransposeConv *>(out);
    if (tconv != nullptr && tconv->bias() == circle_const)
    {
      assert(tconv->outBackprop() != nullptr);
      assert(tconv->filter() != nullptr);
      result.push_back({tconv->outBackprop(), tconv->filter(), tconv});
      continue;
    }
  }
  return result;
}

CircleConst *asym_quant_bias_per_layer(CircleConst *node, float input_scale, float weight_scale,
                                       float *scaling_factor, int64_t *zp)
{
  float scale = input_scale * weight_scale;
  const float scaling_factor_inv = (scale == 0) ? 0 : 1.0 / scale;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    quantized_values[i] =
      static_cast<int32_t>(std::round(node->at<loco::DataType::FLOAT32>(i) * scaling_factor_inv));
  }

  auto new_bias = create_empty_const_from<loco::DataType::S32>(node, size);

  const int32_t kMinScale = std::numeric_limits<int32_t>::lowest();
  const int32_t kMaxScale = std::numeric_limits<int32_t>::max();
  for (uint32_t i = 0; i < size; ++i)
  {
    new_bias->at<loco::DataType::S32>(i) =
      std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
  *scaling_factor = scale;
  *zp = 0;

  return new_bias;
}

CircleConst *quant_bias_per_channel(CircleConst *node, float input_scale,
                                    std::vector<float> &weight_scale,
                                    std::vector<float> &scaling_factor, std::vector<int64_t> &zp)
{
  float scaling_factor_inv{0};

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int32_t> quantized_values(size);

  for (uint32_t i = 0; i < size; ++i)
  {
    scaling_factor[i] = input_scale * weight_scale[i];
    scaling_factor_inv = (scaling_factor[i] == 0) ? 0 : 1.0 / scaling_factor[i];
    quantized_values[i] =
      static_cast<int32_t>(std::round(node->at<loco::DataType::FLOAT32>(i) * scaling_factor_inv));
    zp[i] = 0;
  }

  auto new_bias = create_empty_const_from<loco::DataType::S32>(node, size);

  const int32_t kMinScale = std::numeric_limits<int32_t>::lowest();
  const int32_t kMaxScale = std::numeric_limits<int32_t>::max();
  for (uint32_t i = 0; i < size; ++i)
  {
    new_bias->at<loco::DataType::S32>(i) =
      std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }

  return new_bias;
}

CircleConst *int16_quant_bias_per_channel(CircleConst *node, float input_scale,
                                          std::vector<float> &weight_scale,
                                          std::vector<float> &scaling_factor,
                                          std::vector<int64_t> &zp)
{
  float scaling_factor_inv{0};

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  std::vector<int64_t> quantized_values(size);

  for (uint32_t i = 0; i < size; ++i)
  {
    scaling_factor[i] = input_scale * weight_scale[i];
    scaling_factor_inv = (scaling_factor[i] == 0) ? 0 : 1.0 / scaling_factor[i];
    quantized_values[i] =
      static_cast<int64_t>(std::round(node->at<loco::DataType::FLOAT32>(i) * scaling_factor_inv));
    zp[i] = 0;
  }

  auto new_bias = create_empty_const_from<loco::DataType::S64>(node, size);

  for (uint32_t i = 0; i < size; ++i)
  {
    new_bias->at<loco::DataType::S64>(i) = quantized_values[i];
  }

  return new_bias;
}

bool has_min_max(const CircleNode *node)
{
  return node->quantparam() && !node->quantparam()->min.empty() && !node->quantparam()->max.empty();
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

void set_bias(luci::CircleNode *node, luci::CircleConst *bias)
{
  if (auto conv = dynamic_cast<CircleConv2D *>(node))
    conv->bias(bias);
  else if (auto dconv = dynamic_cast<CircleDepthwiseConv2D *>(node))
    dconv->bias(bias);
  else if (auto tconv = dynamic_cast<CircleTransposeConv *>(node))
    tconv->bias(bias);
  else if (auto fc = dynamic_cast<CircleFullyConnected *>(node))
    fc->bias(bias);
  else
    throw std::runtime_error("Only convolution, depthwise convolution, transposed convolution, and "
                             "fully-connected layer have bias");
}

void set_act_qparam(luci::CircleNode *node, float scale, int64_t zp)
{
  assert(node);               // FIX_CALLER_UNLESS
  assert(node->quantparam()); // FIX_CALLER_UNLESS

  auto qparam = node->quantparam();
  assert(qparam->scale.size() == 1); // FIX_CALLER_UNLESS
  assert(qparam->zerop.size() == 1); // FIX_CALLER_UNLESS
  qparam->scale[0] = scale;
  qparam->zerop[0] = zp;
}

// For nodes with integer output, we use integer scale
void set_int_scale(luci::CircleNode *node)
{
  assert(node); // FIX_CALLER_UNLESS

  auto qparam = node->quantparam();
  assert(qparam);                    // FIX_CALLER_UNLESS
  assert(qparam->scale.size() == 1); // FIX_CALLER_UNLESS

  auto fp_scale = qparam->scale[0];
  qparam->scale[0] = fp_scale < 1 ? 1.0f : std::round(fp_scale);
}

/**
 * @brief Manually set scale/zp of output tensor of special Ops
 */
struct QuantizeSpecialActivation final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeSpecialActivation(loco::DataType input, loco::DataType output)
    : input_type(input), output_type(output)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;

  void visit(luci::CircleNode *node)
  {
    // Nodes fused with activation functions which need special quantization
    auto fused_act_node = dynamic_cast<CircleNodeMixin<CircleNodeTrait::FusedActFunc> *>(node);
    if (fused_act_node != nullptr &&
        fused_act_node->fusedActivationFunction() == FusedActFunc::TANH)
    {
      if (output_type == loco::DataType::U8)
        set_act_qparam(node, 2.0f / 256.0f, 128);
      else
      {
        assert(output_type == loco::DataType::S16);
        set_act_qparam(node, 1.0f / 32768.0f, 0);
      }
    }
  }

  void visit(luci::CircleLogistic *node)
  {
    if (output_type == loco::DataType::U8)
      set_act_qparam(node, 1.0f / 256.0f, 0);
    else
    {
      assert(output_type == loco::DataType::S16);
      set_act_qparam(node, 1.0f / 32768.0f, 0);
    }
  }

  void visit(luci::CircleTanh *node)
  {
    if (output_type == loco::DataType::U8)
      set_act_qparam(node, 2.0f / 256.0f, 128);
    else
    {
      assert(output_type == loco::DataType::S16);
      set_act_qparam(node, 1.0f / 32768.0f, 0);
    }
  }

  void visit(luci::CircleSoftmax *node)
  {
    if (output_type == loco::DataType::U8)
      set_act_qparam(node, 1.0f / 255.0f, 0);
    else
    {
      assert(output_type == loco::DataType::S16);
      set_act_qparam(node, 1.0f / 32767.0f, 0);
    }
  }

  void visit(luci::CircleFloor *node) { set_int_scale(node); }
  void visit(luci::CircleFloorDiv *node) { set_int_scale(node); }
  void visit(luci::CircleFloorMod *node) { set_int_scale(node); }
  void visit(luci::CircleCeil *node) { set_int_scale(node); }
};

/**
 * @brief QuantizeActivation quantizes tensors for activations
 * @details Quantize using recorded min/max values
 */
struct QuantizeActivation final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeActivation(loco::DataType input, loco::DataType output)
    : input_type(input), output_type(output)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;

  // Quantize input tensors of each node
  void visit(luci::CircleNode *node)
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
      auto iwo = get_input_weight_output_of_bias(circle_node);
      if (iwo.size() > 0)
        continue;

      // Check if this is bool type (bool type is not quantized)
      if (circle_node->dtype() == loco::DataType::BOOL)
        continue;

      // Check if this is activation
      // We assume min/max are recorded only for activations
      if (has_min_max(circle_node) && !is_weights(circle_node))
      {
        // Quantize using recorded min/max
        auto quantparam = circle_node->quantparam();
        assert(quantparam);
        assert(quantparam->min.size() == 1); // only support layer-wise quant
        assert(quantparam->max.size() == 1); // only support layer-wise quant
        auto min = quantparam->min[0];
        auto max = quantparam->max[0];

        float scaling_factor{0};
        int64_t zp{0};
        float nudged_min{0};
        float nudged_max{0};

        if (output_type == loco::DataType::U8)
        {
          compute_asym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
          circle_node->dtype(loco::DataType::U8);
        }
        else
        {
          compute_sym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
          circle_node->dtype(loco::DataType::S16);
        }

        circle_node->quantparam()->scale.push_back(scaling_factor);
        circle_node->quantparam()->zerop.push_back(zp);
      }
      // Fix special attributes
      if (circle_node->opcode() == luci::CircleOpcode::CAST)
      {
        auto *cast = loco::must_cast<luci::CircleCast *>(circle_node);
        auto *cast_input = loco::must_cast<luci::CircleNode *>(cast->x());

        // make sure that cast_input is already quantized
        assert(cast_input->dtype() != loco::DataType::FLOAT32);
        cast->in_data_type(cast_input->dtype());
        cast->out_data_type(cast->dtype());
      }
    }
  }
};

struct QuantizeBias final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeBias(loco::DataType input, loco::DataType output, QuantizationGranularity gr)
    : input_type(input), output_type(output), granularity(gr)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;
  QuantizationGranularity granularity;

  // Quantize bias node
  void visit(luci::CircleNode *node)
  {
    // Check if this is already quantized
    if (is_quantized(node))
      return;

    auto iwo_list = get_input_weight_output_of_bias(node);

    for (auto iwo : iwo_list)
    {
      assert(iwo.size() == 3);

      auto input = loco::must_cast<luci::CircleNode *>(iwo[0]);
      auto weight = loco::must_cast<luci::CircleNode *>(iwo[1]);
      auto output = loco::must_cast<luci::CircleNode *>(iwo[2]);

      auto const_bias = loco::must_cast<luci::CircleConst *>(node);
      assert(const_bias->dtype() == loco::DataType::FLOAT32);

      // If input is const, it is quantized here, not in QuantizeActivation
      if (auto const_input = dynamic_cast<luci::CircleConst *>(input))
      {
        quant_const(const_input, output_type);
      }

      CircleConst *new_bias = nullptr;

      if (granularity == QuantizationGranularity::ChannelWise)
      {
        auto input_q = input->quantparam();
        assert(input_q);
        assert(input_q->scale.size() == 1); // input scale's layer-wise
        auto input_scale = input_q->scale[0];

        assert(weight->quantparam() != nullptr); // weight scale's channel-wise
        auto weight_scale = weight->quantparam()->scale;

        uint32_t size = const_bias->size<loco::DataType::FLOAT32>();
        assert(size == weight_scale.size());
        std::vector<float> scaling_factor(size);
        std::vector<int64_t> zp(size);

        if (output_type == loco::DataType::U8)
        {
          new_bias =
            quant_bias_per_channel(const_bias, input_scale, weight_scale, scaling_factor, zp);
        }
        else if (output_type == loco::DataType::S16)
        {
          new_bias =
            int16_quant_bias_per_channel(const_bias, input_scale, weight_scale, scaling_factor, zp);
        }
        else
        {
          throw std::runtime_error("Unsupported quantization type.");
        }

        auto quantparam = std::make_unique<CircleQuantParam>();
        quantparam->scale = scaling_factor;
        quantparam->zerop = zp;
        assert(new_bias->quantparam() == nullptr); // bias should not be quantized before
        new_bias->quantparam(std::move(quantparam));

        set_bias(output, new_bias);
      }
      else
      {
        auto input_q = input->quantparam();
        assert(input_q);
        assert(input_q->scale.size() == 1); // Only support per-layer quant
        auto input_scale = input_q->scale[0];

        auto weight_q = weight->quantparam();
        assert(weight_q);
        assert(weight_q->scale.size() == 1); // Only support per-layer quant
        auto weight_scale = weight_q->scale[0];

        float scaling_factor{0};
        int64_t zp{0};
        new_bias =
          asym_quant_bias_per_layer(const_bias, input_scale, weight_scale, &scaling_factor, &zp);
        auto quantparam = std::make_unique<CircleQuantParam>();
        quantparam->scale.push_back(scaling_factor);
        quantparam->zerop.push_back(zp);
        assert(new_bias->quantparam() == nullptr); // bias should not be quantized before
        new_bias->quantparam(std::move(quantparam));

        set_bias(output, new_bias);
      }
    }
  }
};

/**
 * @brief QuantizeWeights quantizes tensors for weights
 * @details Find min/max values on the fly and then quantize
 */
struct QuantizeWeights final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeWeights(loco::DataType input, loco::DataType output, QuantizationGranularity gr)
    : input_type(input), output_type(output), granularity(gr)
  {
  }

  loco::DataType input_type;
  loco::DataType output_type;
  QuantizationGranularity granularity;

private:
  void quantize_weights(luci::CircleConst *weights)
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

  void visit(luci::CircleConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    if (!is_quantized(weights))
    {
      auto new_weights = luci::clone(weights);
      node->filter(new_weights);
      quantize_weights(new_weights);
    }
  }

  void visit(luci::CircleDepthwiseConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    if (!is_quantized(weights))
    {
      auto new_weights = luci::clone(weights);
      node->filter(new_weights);
      quantize_weights(new_weights);
    }
  }

  void visit(luci::CircleInstanceNorm *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

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

  void visit(luci::CirclePRelu *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

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

  void visit(luci::CircleTransposeConv *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    if (!is_quantized(weights))
    {
      auto new_weights = luci::clone(weights);
      node->filter(new_weights);
      quantize_weights(new_weights);
    }
  }

  void visit(luci::CircleFullyConnected *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

    auto weights = loco::must_cast<luci::CircleConst *>(node->weights());
    if (!is_quantized(weights))
    {
      auto new_weights = luci::clone(weights);
      node->weights(new_weights);
      quantize_weights(new_weights);
    }
  }

  void visit(luci::CircleNode *) {}
};

/** EXAMPLE
 *
 * BEFORE
 *
 *         [CircleNode]       [CircleConst]
 *           (qparam1)           (FP32)
 *                   \            /
 *                    \          /
 *                    [CirclePack]
 *                     (qparam2)
 *
 *  AFTER
 *
 *         [CircleNode]        [CircleConst]   [CircleConst] <- Dead node
 *           (qparam2)           (qparam2)         (FP32)
 *                   \            /
 *                    \          /
 *                    [CirclePack]
 *                     (qparam2)
 *
 * NOTE Quantization parameter of CirclePack (qparam2) is propagated to the inputs.
 */
void propagate_pack_quantparam(luci::CirclePack *pack, loco::DataType quant_type)
{
  assert(pack->quantparam() != nullptr);

  const auto num_inputs = pack->values_count();

  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto node = loco::must_cast<luci::CircleNode *>(pack->arg(i));

    // Skip if this input is PACK Op
    if (node->opcode() == luci::CircleOpcode::PACK)
      continue;

    // Quantize constant values
    if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    {
      luci::CircleConst *const_node = loco::must_cast<luci::CircleConst *>(node);
      if (const_node->dtype() != loco::DataType::FLOAT32)
        throw std::runtime_error("Unsupported data type for constant input of pack Op");

      const auto pack_qparam = pack->quantparam();
      if (pack_qparam == nullptr)
        throw std::runtime_error("quantparam of pack is not found during propagation");

      assert(pack_qparam->scale.size() == 1);
      assert(pack_qparam->zerop.size() == 1);
      const auto scaling_factor = pack_qparam->scale[0];
      const auto zerop = pack_qparam->zerop[0];

      auto new_const = luci::clone(const_node);
      quant_const_values(new_const, scaling_factor, zerop, quant_type);
      pack->values(i, new_const);
      overwrite_quantparam(pack, new_const);
    }
    else
    {
      const auto succs = loco::succs(node);
      if (succs.size() > 1)
        continue;

      // Non-const input must have been quantized
      assert(node->quantparam() != nullptr);
      overwrite_quantparam(pack, node);
    }
  }
}

/** EXAMPLE
 *
 *
 *
 * BEFORE
 *
 *      [CircleNode] [CircleConst] [CircleConst] [CircleNode]
 *          (S32)        (S32)        (FP32)     (U8 qparam1)
 *              \          \           /            /
 *               \          \        /            /
 *                \          \     /            /
 *                 -------[CircleOneHot]-------
 *                         (U8 qparam2)
 *
 *  AFTER
 *
 *      [CircleNode] [CircleConst] [CircleConst] [CircleNode]      [CircleConst] <- Dead node
 *          (S32)        (S32)     (U8 qparam2)  (U8 qparam2)         (FP32)
 *              \          \           /           /
 *               \          \        /            /
 *                \          \     /            /
 *                 -------[CircleOneHot]-------
 *                         (U8 qparam2)
 *
 * NOTE Quantization parameter of CircleOneHot (qparam2) is propagated to on_value/off_value.
 */
void propagate_one_hot_quantparam(luci::CircleOneHot *one_hot, loco::DataType quant_type)
{
  assert(one_hot->quantparam() != nullptr);

  // Propagate quantization parameters from output to inputs,
  // to fit both input and counstant_value in one quant range.
  auto quant_input = [one_hot, quant_type](void (CircleOneHot::*arg_setter)(loco::Node *),
                                           loco::Node *(CircleOneHot::*arg_getter)() const) {
    auto node = loco::must_cast<luci::CircleNode *>((one_hot->*arg_getter)());

    // Quantize constant values
    if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    {
      luci::CircleConst *const_node = loco::must_cast<luci::CircleConst *>(node);
      if (is_quantized(const_node))
        return;

      if (const_node->dtype() != loco::DataType::FLOAT32)
        throw std::runtime_error("Unsupported data type for constant input of OneHot Op");

      const auto qparam = one_hot->quantparam();
      if (qparam == nullptr)
        throw std::runtime_error("quantparam of OneHot is not found during propagation");

      assert(qparam->scale.size() == 1);
      const auto scaling_factor = qparam->scale.at(0);
      const auto zerop = qparam->zerop.at(0);

      auto new_const = luci::clone(const_node);
      quant_const_values(new_const, scaling_factor, zerop, quant_type);
      overwrite_quantparam(one_hot, new_const);
      (one_hot->*arg_setter)(new_const);
    }
    // Subsequent OneHot Ops quant params are not propagated
    else if (node->opcode() == luci::CircleOpcode::ONE_HOT)
    {
      return;
    }
    else
    {
      const auto succs = loco::succs(node);
      if (succs.size() > 1)
        return;

      // Non-const input must have been quantized
      assert(node->quantparam() != nullptr);
      overwrite_quantparam(one_hot, node);
    }
  };

  quant_input(&CircleOneHot::on_value, &CircleOneHot::on_value);
  quant_input(&CircleOneHot::off_value, &CircleOneHot::off_value);
}

// Quantize constant input activation of a node
// The input of a node is quantized if it is
// 1. Constant (instance of CircleConst*)
// 2. Activation (other inputs e.g., weights, bias, axis, etc should not be quantized here)
struct QuantizeConstInputActivation final : public luci::CircleNodeMutableVisitor<void>
{
  QuantizeConstInputActivation(loco::DataType output_type) : _output_type(output_type) {}

private:
  loco::DataType _output_type;

// Skip NODE
#define SKIP(NODE) \
  void visit(NODE *) {}

// INPUT_NAME is the only activation of NODE
#define QUANTIZE_SINGLE_CONST_INPUT(NODE, INPUT_NAME)           \
  void visit(NODE *node)                                        \
  {                                                             \
    auto input = node->INPUT_NAME();                            \
    auto const_node = dynamic_cast<luci::CircleConst *>(input); \
    if (const_node && !is_quantized(const_node))                \
    {                                                           \
      auto new_const = luci::clone(const_node);                 \
      quant_const(new_const, _output_type);                     \
      node->INPUT_NAME(new_const);                              \
    }                                                           \
  }

// INPUT_NAME1 and INPUT_NAME2 are the only activations of NODE
#define QUANTIZE_TWO_CONST_INPUTS(NODE, INPUT_NAME1, INPUT_NAME2) \
  void visit(NODE *node)                                          \
  {                                                               \
    auto input1 = node->INPUT_NAME1();                            \
    auto const_node1 = dynamic_cast<luci::CircleConst *>(input1); \
    if (const_node1 && !is_quantized(const_node1))                \
    {                                                             \
      auto new_const1 = luci::clone(const_node1);                 \
      quant_const(new_const1, _output_type);                      \
      node->INPUT_NAME1(new_const1);                              \
    }                                                             \
    auto input2 = node->INPUT_NAME2();                            \
    auto const_node2 = dynamic_cast<luci::CircleConst *>(input2); \
    if (const_node2 && !is_quantized(const_node2))                \
    {                                                             \
      auto new_const2 = luci::clone(const_node2);                 \
      quant_const(new_const2, _output_type);                      \
      node->INPUT_NAME2(new_const2);                              \
    }                                                             \
  }

  // Default behavior (NYI)
  void visit(luci::CircleNode *node)
  {
    for (uint32_t i = 0; i < node->arity(); i++)
    {
      auto input_node = node->arg(i);
      auto const_node = dynamic_cast<luci::CircleConst *>(input_node);
      if (const_node != nullptr)
        throw std::runtime_error("Unsupported Op for const inputs");
    }
  }

  // Handled in QuantizeWeights and QuantizeBias
  SKIP(luci::CircleConv2D)
  SKIP(luci::CircleDepthwiseConv2D)
  SKIP(luci::CircleFullyConnected)
  SKIP(luci::CircleInstanceNorm)
  SKIP(luci::CirclePRelu)
  SKIP(luci::CircleTransposeConv)

  // Handled in PropagateQParamBackwardPass
  SKIP(luci::CircleConcatenation)

  // Inputs of logical Ops are bool, thus not quantized
  SKIP(luci::CircleLogicalOr)
  SKIP(luci::CircleLogicalAnd)
  SKIP(luci::CircleLogicalNot)

  // Ops that receive a single activation as an input
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleArgMax, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleArgMin, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleBatchToSpaceND, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleDepthToSpace, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleElu, features)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleExp, x)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleFloor, x)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleGather, params)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleLocalResponseNormalization, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleLogistic, x)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleMean, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleMirrorPad, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CirclePad, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleReduceAny, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleReduceProd, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleReduceMax, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleReduceMin, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleReshape, tensor)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleResizeBilinear, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleResizeNearestNeighbor, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleReverseSequence, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleRsqrt, x)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleSlice, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleSoftmax, logits)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleSpaceToBatchND, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleSpaceToDepth, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleSplit, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleSplitV, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleSqrt, x)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleStridedSlice, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleSum, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleTanh, x)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleTile, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleTopKV2, input)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleTranspose, a)
  QUANTIZE_SINGLE_CONST_INPUT(luci::CircleUnpack, value)

  // Ops that receive two activations as inputs
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleAdd, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleBatchMatMul, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleDiv, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleEqual, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleFloorDiv, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleGreater, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleGreaterEqual, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleLess, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleLessEqual, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleMaximum, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleMinimum, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleMul, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleNotEqual, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CirclePow, x, y)
  QUANTIZE_TWO_CONST_INPUTS(luci::CircleSub, x, y)

  // AddN has arbitrary number of inputs
  void visit(luci::CircleAddN *node)
  {
    auto arity = node->arity();
    for (uint32_t i = 0; i < arity; i++)
    {
      auto input_node = node->inputs(i);
      auto const_node = dynamic_cast<luci::CircleConst *>(input_node);
      if (const_node && !is_quantized(const_node))
      {
        auto new_const = luci::clone(const_node);
        quant_const(new_const, _output_type);
        node->inputs(i, new_const);
      }
    }
  }

  // Ops whose input qparam is determined by output qparam
  void visit(luci::CirclePadV2 *node) { propagate_pad_v2_quantparam(node, _output_type); }
  void visit(luci::CirclePack *node) { propagate_pack_quantparam(node, _output_type); }
  void visit(luci::CircleOneHot *node) { propagate_one_hot_quantparam(node, _output_type); }

#undef SKIP
#undef QUANTIZE_SINGLE_CONST_INPUT
#undef QUANTIZE_TWO_CONST_INPUTS
};

} // namespace

/**
 * tells if pad_v2 quantization should ignore padding value
 * In that case padding const will be quantized with input parameters, and probably clipped
 */
bool ignore_pad_v2_const_quantization(luci::CirclePadV2 *pad)
{
  // This is a workaround to quantize pad generated from MaxPoolWithArgmax operation properly
  // TODO use metadata hints to detect this case
  auto const_value_node = dynamic_cast<luci::CircleConst *>(pad->arg(2));
  if (!const_value_node)
    return false;
  if (const_value_node->dtype() == loco::DataType::FLOAT32)
  {
    float const_value = const_value_node->at<loco::DataType::FLOAT32>(0);
    if (const_value == std::numeric_limits<float>::lowest())
      return true;
  }
  return false;
}

/** BEFORE
 *
 *         [CircleNode] [CircleConst] [CircleConst]
 *         (U8 qparam1)     (S32)       (FP32)
 *                   \        |         /
 *                    \       |        /
 *                      [CirclePadV2]
 *                       (U8 qparam2)
 *
 *  AFTER (case 1)
 *
 *  By default qparam is propagated from output to inputs to meet backend requirements.
 *
 *         [CircleNode] [CircleConst] [CircleConst]   [CircleConst] <- Dead node
 *         (U8 qparam2)     (S32)      (U8 qparam2)       (FP32)
 *                   \        |         /
 *                    \       |        /
 *                      [CirclePadV2]
 *                       (U8 qparam2)
 *
 *  AFTER (case 2)
 *
 * In case padded value is the lowest float value
 * Qparam is propagated from input to output and constant.
 *
 * This is a special case for optimization constructed pad, needed to guarantee that
 * extremely large negative constant do not stretch output quantization range.
 *
 *         [CircleNode] [CircleConst] [CircleConst]   [CircleConst] <- Dead node
 *         (U8 qparam1)     (S32)      (U8 qparam1)       (FP32)
 *                   \        |         /
 *                    \       |        /
 *                      [CirclePadV2]
 *                       (U8 qparam1)
 */
void propagate_pad_v2_quantparam(luci::CirclePadV2 *pad_v2, loco::DataType quant_type)
{
  if (ignore_pad_v2_const_quantization(pad_v2))
  {
    // propagate input quantization paramters from input to output and padding const value
    auto pad_v2_input = loco::must_cast<luci::CircleNode *>(pad_v2->arg(0));
    overwrite_quantparam(pad_v2_input, pad_v2);

    auto const_value_node = loco::must_cast<luci::CircleConst *>(
      pad_v2->arg(2)); // FIX ignore_pad_v2_const_quantization UNLESS
    auto new_const = luci::clone(const_value_node);

    const auto pad_v2_input_qparam = pad_v2_input->quantparam();
    assert(pad_v2_input_qparam != nullptr);
    assert(pad_v2_input_qparam->scale.size() == 1);
    const auto scaling_factor = pad_v2_input_qparam->scale.at(0);
    const auto zerop = pad_v2_input_qparam->zerop.at(0);

    quant_const_values(new_const, scaling_factor, zerop, quant_type);
    overwrite_quantparam(pad_v2_input, new_const);
    pad_v2->constant_values(new_const);
    return;
  }

  // Propagate quantization paramters from output to inputs,
  // to fit both input and counstant_value in one quant range.
  auto quant_input = [pad_v2, quant_type](void (CirclePadV2::*arg_setter)(loco::Node *),
                                          uint32_t arg) {
    auto node = loco::must_cast<luci::CircleNode *>(pad_v2->arg(arg));

    // Quantize constant values
    if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    {
      luci::CircleConst *const_node = loco::must_cast<luci::CircleConst *>(node);
      if (is_quantized(const_node))
        return;

      if (const_node->dtype() != loco::DataType::FLOAT32)
        throw std::runtime_error("Unsupported data type for constant input of PadV2 Op");

      const auto pad_v2_qparam = pad_v2->quantparam();
      if (pad_v2_qparam == nullptr)
        throw std::runtime_error("quantparam of PadV2 is not found during propagation");

      assert(pad_v2_qparam->scale.size() == 1);
      const auto scaling_factor = pad_v2_qparam->scale.at(0);
      const auto zerop = pad_v2_qparam->zerop.at(0);

      auto new_const = luci::clone(const_node);
      quant_const_values(new_const, scaling_factor, zerop, quant_type);
      overwrite_quantparam(pad_v2, new_const);
      (pad_v2->*arg_setter)(new_const);
    }
    // Subsequent PadV2 Ops quant params are not propagated
    else if (node->opcode() == luci::CircleOpcode::PADV2)
    {
      return;
    }
    else
    {
      const auto succs = loco::succs(node);
      if (succs.size() > 1)
        return;

      // Non-const input must have been quantized
      assert(node->quantparam() != nullptr);
      overwrite_quantparam(pad_v2, node);
    }
  };

  quant_input(&CirclePadV2::input, 0);
  quant_input(&CirclePadV2::constant_values, 2);
}

void QuantizeWithMinMaxPass::set_input_type(loco::Graph *g) const
{
  auto inputs = g->inputs();
  for (auto node : loco::input_nodes(g))
  {
    auto input = loco::must_cast<luci::CircleInput *>(node);
    if (input->dtype() == _ctx->input_type)
      continue;

    // Bool type is not quantizable
    if (input->dtype() == loco::DataType::BOOL)
      continue;
    if (input->dtype() == loco::DataType::S32)
      continue;
    if (input->dtype() == loco::DataType::S64)
      continue;

    // Insert Quantize Op
    auto quant_op = create_quantize_op(input, input->dtype());
    loco::replace(input).with(quant_op);
    quant_op->input(input);

    // TODO Set a proper origin (Quantize should have its own Origin)
    {
      auto succs = loco::succs(quant_op);
      assert(succs.size() > 0);
      auto succ = loco::must_cast<luci::CircleNode *>(*succs.begin());
      luci::add_origin(quant_op, luci::get_origin(succ));
    }

    // Requantize input
    {
      auto quantparam = input->quantparam();
      assert(quantparam);
      assert(quantparam->min.size() == 1); // only support layer-wise quant
      assert(quantparam->max.size() == 1); // only support layer-wise quant
      auto min = quantparam->min[0];
      auto max = quantparam->max[0];

      float scaling_factor{0};
      int64_t zp{0};
      float nudged_min{0};
      float nudged_max{0};

      if (_ctx->input_type == loco::DataType::U8)
      {
        compute_asym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
      }
      else
      {
        assert(_ctx->input_type == loco::DataType::S16);
        compute_sym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
      }
      input->dtype(_ctx->input_type);
      input->quantparam()->scale[0] = scaling_factor;
      input->quantparam()->zerop[0] = zp;
    }

    auto graph_input = inputs->at(input->index());
    graph_input->dtype(_ctx->input_type);
  }
}

void QuantizeWithMinMaxPass::set_output_type(loco::Graph *g) const
{
  auto outputs = g->outputs();
  for (auto node : loco::output_nodes(g))
  {
    auto output = loco::must_cast<luci::CircleOutput *>(node);
    if (output->dtype() == _ctx->output_type)
      continue;

    // Bool type is not quantizable
    if (output->dtype() == loco::DataType::BOOL)
      continue;

    auto from = loco::must_cast<luci::CircleNode *>(output->from());

    // The last Op is not quantizable Op (ex: ArgMax)
    if (not from->quantparam())
      continue;

    // Insert Quantize Op
    auto quant_op = create_quantize_op(from, _ctx->output_type);
    loco::replace(from).with(quant_op);
    quant_op->input(from);

    // TODO Set a proper origin (Quantize should have its own Origin)
    luci::add_origin(quant_op, luci::get_origin(from));

    auto graph_output = outputs->at(output->index());
    graph_output->dtype(_ctx->output_type);
  }
}

/**
 * How QuantizeWithMinMax works?
 *
 * We categorized tensors into four groups
 * - Activation: Feature maps (both Const/Non-const)
 * - Weights: Const tensors of specific Ops (Conv, FC, ...)
 * - Bias: Const tensors of specific Ops (Conv, FC, ...)
 * - Others: padding value, one_hot value, axis, ..
 *
 * Activation is quantized in different ways
 * 1. For non-constant activation, quantize using recorded min/max
 * 2. For constant activation, quantize using min/max of its value
 * 3. For some Ops (ex: pad_v2), output qparam is used as input qparam (backward propagation)
 * 4. For some Ops (ex: reshape), input qparam is used as output qparam (forward propagation)
 * 5. For some Ops (ex: tanh), output qparam has pre-defined values
 *
 * Weights is quantized using min/max of its value
 *
 * Bias is quantized using input scale (s_i) and weights scale (s_w)
 * - Activation and weights should be quantized earlier than bias
 *
 * Quantization Steps
 * 1. Quantize Activation
 *   - Quantize using recorded min/max (QuantizeActivation)
 *   - Propagate qparam backward (PropagateQParamBackwardPass)
 *   - Quantize const inputs + propagate qparam backward (QuantizeConstInputActivation)
 *   - Quantize using pre-defined values (QuantizeSpecialActivation)
 *   - Propagate qparam forward (PropagateQParamForwardPass)
 * 2. Quantize Weights
 * 3. Quantize Bias
 * 4. Set input dtype
 * 5. Set output dtype
 *
 * Why quantization sequence was determined as above?
 * - Activation and weights should be quantized before bias (1->2->3). Input/Output
 *   dtype can be updated at the end (4->5).
 * - During activation quantization,
 *   - Backward propagation is performed earlier than forward propagation. This allows
 *     backward-propagated qpram to be overwritten during forward propagation.
 *     We made this decision as Ops for forward propagation (reshape, transpose, ..)
 *     are more common than backward propagation. TODO Check this decision is safe.
 *   - QuantizeSpecialActivation is called before forward propagation to make sure that
 *     the pre-defined qparam values are propagated.
 */
bool QuantizeWithMinMaxPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "QuantizeWithMinMaxPass Start" << std::endl;

  // Quantize activation
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    QuantizeActivation qa(_ctx->input_model_dtype, _ctx->output_model_dtype);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qa);
  }

  // Backward propagation of activation qparam
  {
    PropagateQParamBackwardPass pqbp(_ctx->output_model_dtype);
    pqbp.run(g);
  }

  // Quantize const input activation
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    QuantizeConstInputActivation qcia(_ctx->output_model_dtype);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qcia);
  }

  // Update qparam of output of special Ops
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    QuantizeSpecialActivation qsa(_ctx->input_model_dtype, _ctx->output_model_dtype);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qsa);
  }

  // Forward propagation of activation qparam
  logo::Phase phase;

  phase.emplace_back(std::make_unique<luci::PropagateQParamForwardPass>(_ctx->TF_style_maxpool));

  ProgressReporter prog(g, logo::PhaseStrategy::Saturate);
  logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);

  // Quantize weights
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    QuantizeWeights qw(_ctx->input_model_dtype, _ctx->output_model_dtype, _ctx->granularity);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qw);
  }

  // Quantize bias
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    QuantizeBias qb(_ctx->input_model_dtype, _ctx->output_model_dtype, _ctx->granularity);
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    circle_node->accept(&qb);
  }

  // Update output dtype
  auto graph_outputs = g->outputs();
  for (auto node : loco::output_nodes(g))
  {
    auto circle_node = loco::must_cast<luci::CircleOutput *>(node);
    if (static_cast<luci::CircleNode *>(circle_node->from())->dtype() == _ctx->output_model_dtype)
    {
      circle_node->dtype(_ctx->output_model_dtype);
      auto graph_output = graph_outputs->at(circle_node->index());
      graph_output->dtype(_ctx->output_model_dtype);
    }
  }

  // Set input type
  set_input_type(g);

  // Set output type
  set_output_type(g);

  // Remove min/max values
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (auto qparam = circle_node->quantparam())
    {
      qparam->min.clear();
      qparam->max.clear();
    }
  }

  INFO(l) << "QuantizeWithMinMaxPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
