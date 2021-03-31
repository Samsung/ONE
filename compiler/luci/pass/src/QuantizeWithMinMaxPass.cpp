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
#include "QuantizationUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>

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

void overwrite_quantparam(luci::CircleConcatenation *concat, luci::CircleNode *target)
{
  auto concat_qparam = concat->quantparam();
  if (concat_qparam == nullptr)
    throw std::runtime_error("quantparam of concat is not found during overwrite");

  auto target_qparam = target->quantparam();
  if (target_qparam == nullptr)
  {
    auto quantparam = std::make_unique<CircleQuantParam>();
    target->quantparam(std::move(quantparam));
    target_qparam = target->quantparam();

    if (target_qparam == nullptr)
      throw std::runtime_error("Creating new quant param failed");
  }
  target_qparam->min = concat_qparam->min;
  target_qparam->max = concat_qparam->max;
  target_qparam->scale = concat_qparam->scale;
  target_qparam->zerop = concat_qparam->zerop;
  target_qparam->quantized_dimension = concat_qparam->quantized_dimension;
}

void quant_const_values(luci::CircleConst *const_node, float scaling_factor, float zerop,
                        loco::DataType quant_type)
{
  uint32_t size = const_node->size<loco::DataType::FLOAT32>();

  const float scaling_factor_inv = 1.0 / scaling_factor;
  std::vector<int32_t> quantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    auto data = const_node->at<loco::DataType::FLOAT32>(i);
    quantized_values[i] = static_cast<int32_t>(std::round(data * scaling_factor_inv) + zerop);
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
      auto iwo = get_input_weight_output_of_bias(circle_node);
      if (iwo.size() > 0)
        continue;

      // Check if this is activation
      // We assume min/max are recorded only for activations
      if (has_min_max(circle_node) && !is_weights(circle_node))
      {
        // Quantize using recorded min/max
        auto quantparam = circle_node->quantparam();
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

        circle_node->quantparam()->min.clear();
        circle_node->quantparam()->max.clear();
        circle_node->quantparam()->scale.push_back(scaling_factor);
        circle_node->quantparam()->zerop.push_back(zp);
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

    auto iwo_list = get_input_weight_output_of_bias(node);

    for (auto iwo : iwo_list)
    {
      assert(iwo.size() == 3);

      auto input = loco::must_cast<luci::CircleNode *>(iwo[0]);
      auto weight = loco::must_cast<luci::CircleNode *>(iwo[1]);
      auto output = loco::must_cast<luci::CircleNode *>(iwo[2]);

      auto const_bias = loco::must_cast<luci::CircleConst *>(node);
      assert(const_bias->dtype() == loco::DataType::FLOAT32);

      CircleConst *new_bias = nullptr;

      if (granularity == QuantizationGranularity::ChannelWise)
      {
        assert(input->quantparam()->scale.size() == 1); // input scale's layer-wise
        auto input_scale = input->quantparam()->scale[0];

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
        assert(input->quantparam()->scale.size() == 1); // Only support per-layer quant
        auto input_scale = input->quantparam()->scale[0];

        assert(weight->quantparam()->scale.size() == 1); // Only support per-layer quant
        auto weight_scale = weight->quantparam()->scale[0];

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

  bool visit(luci::CircleConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    if (!is_quantized(weights))
    {
      auto new_weights = luci::clone(weights);
      node->filter(new_weights);
      quantize_weights(new_weights);
      return true;
    }
    return false;
  }

  bool visit(luci::CircleDepthwiseConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    if (!is_quantized(weights))
    {
      auto new_weights = luci::clone(weights);
      node->filter(new_weights);
      quantize_weights(new_weights);
      return true;
    }
    return false;
  }

  bool visit(luci::CircleInstanceNorm *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

    auto gamma = loco::must_cast<luci::CircleConst *>(node->gamma());
    auto beta = loco::must_cast<luci::CircleConst *>(node->beta());

    bool changed = false;
    if (!is_quantized(gamma))
    {
      assert(gamma->dtype() == loco::DataType::FLOAT32);
      auto new_gamma = luci::clone(gamma);
      if (granularity == QuantizationGranularity::LayerWise)
        quant_const(new_gamma, output_type);
      else if (granularity == QuantizationGranularity::ChannelWise)
        quant_const_per_channel(new_gamma, output_type);
      node->gamma(new_gamma);
      changed = true;
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
      changed = true;
    }

    return changed;
  }

  bool visit(luci::CirclePRelu *node)
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
      return true;
    }

    return false;
  }

  bool visit(luci::CircleTransposeConv *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

    auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
    if (!is_quantized(weights))
    {
      auto new_weights = luci::clone(weights);
      node->filter(new_weights);
      quantize_weights(new_weights);
      return true;
    }
    return false;
  }

  bool visit(luci::CircleFullyConnected *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeWeights visit node: " << node->name() << std::endl;

    auto weights = loco::must_cast<luci::CircleConst *>(node->weights());
    if (!is_quantized(weights))
    {
      auto new_weights = luci::clone(weights);
      node->weights(new_weights);
      quantize_weights(new_weights);
      return true;
    }
    return false;
  }

  bool visit(luci::CircleNode *) { return false; }
};

/**
 * @brief Quantize const input tensors using min/max of const values
 */
void quantize_const_inputs(luci::CircleNode *node, loco::DataType output_type)
{
  auto opcode = node->opcode();
  auto arity = node->arity();

  loco::Node *input_node{nullptr};
  luci::CircleConst *const_node{nullptr};

  switch (opcode)
  {
    case luci::CircleOpcode::CONV_2D:
    case luci::CircleOpcode::DEPTHWISE_CONV_2D:
    case luci::CircleOpcode::FULLY_CONNECTED:
    case luci::CircleOpcode::INSTANCE_NORM:
    case luci::CircleOpcode::PRELU:
    case luci::CircleOpcode::TRANSPOSE_CONV:
      // Handled in QuantizeWeights and QuantizeBias
      break;

    case luci::CircleOpcode::CONCATENATION:
      // Handled in propagate_concat_quantparam
      break;

    case luci::CircleOpcode::ARG_MAX:
    case luci::CircleOpcode::ARG_MIN:
    case luci::CircleOpcode::MEAN:
    case luci::CircleOpcode::PAD:
    case luci::CircleOpcode::REDUCE_ANY:
    case luci::CircleOpcode::REDUCE_PROD:
    case luci::CircleOpcode::REDUCE_MAX:
    case luci::CircleOpcode::REDUCE_MIN:
    case luci::CircleOpcode::RESHAPE:
    case luci::CircleOpcode::RESIZE_BILINEAR:
    case luci::CircleOpcode::RESIZE_NEAREST_NEIGHBOR:
    case luci::CircleOpcode::REVERSE_SEQUENCE:
    case luci::CircleOpcode::SUM:
    case luci::CircleOpcode::TILE:
    case luci::CircleOpcode::TOPK_V2:
    case luci::CircleOpcode::TRANSPOSE:
      // The second input of these Ops should not be quantized
      // Ex: axis, paddings
      input_node = node->arg(0);
      const_node = dynamic_cast<luci::CircleConst *>(input_node);
      if (const_node != nullptr && !is_quantized(const_node))
        quant_const(const_node, output_type);
      break;

    case luci::CircleOpcode::ADD:
    case luci::CircleOpcode::ADD_N:
    case luci::CircleOpcode::DIV:
    case luci::CircleOpcode::EQUAL:
    case luci::CircleOpcode::GREATER:
    case luci::CircleOpcode::GREATER_EQUAL:
    case luci::CircleOpcode::LESS:
    case luci::CircleOpcode::LESS_EQUAL:
    case luci::CircleOpcode::MAXIMUM:
    case luci::CircleOpcode::MINIMUM:
    case luci::CircleOpcode::MUL:
    case luci::CircleOpcode::NOT_EQUAL:
    case luci::CircleOpcode::SUB:
      // Quantize all const inputs using their values
      for (uint32_t i = 0; i < arity; i++)
      {
        input_node = node->arg(i);
        const_node = dynamic_cast<luci::CircleConst *>(input_node);
        if (const_node != nullptr && !is_quantized(const_node))
          quant_const(const_node, output_type);
      }
      break;

    case luci::CircleOpcode::SPLIT:
      // Only the second input is quantized
      // First input should not be quantized (e.g., split_dim)
      input_node = node->arg(1);
      const_node = dynamic_cast<luci::CircleConst *>(input_node);
      if (const_node != nullptr && !is_quantized(const_node))
        quant_const(const_node, output_type);
      break;

    default:
      for (uint32_t i = 0; i < arity; i++)
      {
        input_node = node->arg(i);
        const_node = dynamic_cast<luci::CircleConst *>(input_node);
        if (const_node != nullptr)
          throw std::runtime_error("Unsupported Op for const inputs");
      }
      break;
  }
}

} // namespace

/** BEFORE
 *
 *         [CircleNode]             [CircleConst]
 *         (U8 qparam1)                 (FP32)
 *                   \                    /
 *                    \                  /
 *                    [CircleConcatenation]
 *                        (U8 qparam2)
 *
 *  AFTER
 *         [CircleNode]             [CircleConst]   [CircleConst] <- Dead node
 *         (U8 qparam2)             (U8 qparam2)       (FP32)
 *                   \                    /
 *                    \                  /
 *                    [CircleConcatenation]
 *                        (U8 qparam2)
 */
void propagate_concat_quantparam(luci::CircleConcatenation *concat, loco::DataType quant_type)
{
  assert(concat->quantparam() != nullptr);

  const auto num_inputs = concat->numValues();

  // Quantize const inputs using their values if concat has fused act function
  if (concat->fusedActivationFunction() != luci::FusedActFunc::NONE)
  {
    for (uint32_t i = 0; i < num_inputs; i++)
    {
      auto node = concat->arg(i);
      auto const_node = dynamic_cast<luci::CircleConst *>(node);
      if (const_node != nullptr)
      {
        auto new_const = luci::clone(const_node);
        quant_const(new_const, quant_type);
        concat->values(i, new_const);
      }
    }
    return;
  }

  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto node = loco::must_cast<luci::CircleNode *>(concat->arg(i));

    // Skip if this input is CONCAT Op
    if (node->opcode() == luci::CircleOpcode::CONCATENATION)
      continue;

    // Quantize constant values
    if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    {
      luci::CircleConst *const_node = loco::must_cast<luci::CircleConst *>(node);
      if (const_node->dtype() != loco::DataType::FLOAT32)
        throw std::runtime_error("Unsupported data type for constant input of concatenation Op");

      const auto concat_qparam = concat->quantparam();
      if (concat_qparam == nullptr)
        throw std::runtime_error("quantparam of concat is not found during propagation");

      assert(concat_qparam->scale.size() == 1);
      const auto scaling_factor = concat_qparam->scale[0];
      const auto zerop = concat_qparam->zerop[0];

      auto new_const = luci::clone(const_node);
      quant_const_values(new_const, scaling_factor, zerop, quant_type);
      concat->values(i, new_const);
      overwrite_quantparam(concat, new_const);
    }
    else
    {
      const auto succs = loco::succs(node);
      if (succs.size() > 1)
        continue;

      // Non-const input must have been quantized
      assert(node->quantparam() != nullptr);
      overwrite_quantparam(concat, node);
    }
  }
}

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

  // Propagate quantization parameters of concat Op
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto concat = dynamic_cast<luci::CircleConcatenation *>(node);
    if (not concat)
      continue;

    // Propagate qparam of concat to its inputs if
    // (1) concat is uint8-quantized
    // (2) concat has no fused activation function
    // (3) the input is not concatenation Op
    // (4) the input is not produced to Ops other than concat
    propagate_concat_quantparam(concat, _output_dtype);
  }

  // Quantize const inputs other than weights and bias
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    quantize_const_inputs(circle_node, _output_dtype);
  }

  // Update output dtype
  auto graph_outputs = g->outputs();
  for (auto node : loco::output_nodes(g))
  {
    auto circle_node = loco::must_cast<luci::CircleOutput *>(node);
    if (static_cast<luci::CircleNode *>(circle_node->from())->dtype() == _output_dtype)
    {
      circle_node->dtype(_output_dtype);
      auto graph_output = graph_outputs->at(circle_node->index());
      graph_output->dtype(_output_dtype);
    }
  }

  INFO(l) << "QuantizeWithMinMaxPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
