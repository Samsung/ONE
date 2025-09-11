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

#include "QuantizeBias.h"
#include "QuantizationUtils.h"

#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace luci;

namespace
{

// struct to carry Input/Weights/Bias
struct IWB
{
  CircleNode *input = nullptr;
  CircleNode *weights = nullptr;
  CircleConst *bias = nullptr;

  IWB(loco::Node *i, loco::Node *w, loco::Node *b)
  {
    input = dynamic_cast<luci::CircleNode *>(i);
    weights = dynamic_cast<luci::CircleNode *>(w);
    bias = dynamic_cast<luci::CircleConst *>(b);
  }

  // Return true if bias can be quantized with valid input an weights
  operator bool()
  {
    if (bias == nullptr || is_quantized(bias))
      return false;
    if (input == nullptr || weights == nullptr)
      return false;
    return true;
  }
};

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

} // namespace

namespace luci
{

// Return a quantized bias node
CircleConst *QuantizeBias::quantized_bias(CircleNode *input, const CircleNode *weight,
                                          CircleNode *bias)
{
  auto const_bias = luci::must_cast<luci::CircleConst *>(bias);
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

    if (const_bias->rank() == 0)
    {
      // TODO Support quantization of scalar bias
      throw std::runtime_error("Quantization of scalar bias is not yet supported (" +
                               const_bias->name() + ")");
    }
    if (size != const_bias->dim(const_bias->rank() - 1).value())
    {
      throw std::runtime_error(const_bias->name() +
                               " (bias) should have the shape of [1, 1, .. 1, channel]");
    }

    if (output_type == loco::DataType::U8)
    {
      new_bias = quant_bias_per_channel(const_bias, input_scale, weight_scale, scaling_factor, zp);
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
    quantparam->quantized_dimension = const_bias->rank() - 1;
    assert(new_bias->quantparam() == nullptr); // bias should not be quantized before
    new_bias->quantparam(std::move(quantparam));

    return new_bias;
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

    return new_bias;
  }
}

void QuantizeBias::visit(luci::CircleConv2D *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeBias QuantizeBias::visit node: " << node->name() << std::endl;

  if (auto iwb = IWB(node->input(), node->filter(), node->bias()))
  {
    auto new_bias = quantized_bias(iwb.input, iwb.weights, iwb.bias);
    node->bias(new_bias);
  }
}

void QuantizeBias::visit(luci::CircleDepthwiseConv2D *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeBias QuantizeBias::visit node: " << node->name() << std::endl;

  if (auto iwb = IWB(node->input(), node->filter(), node->bias()))
  {
    auto new_bias = quantized_bias(iwb.input, iwb.weights, iwb.bias);
    node->bias(new_bias);
  }
}

void QuantizeBias::visit(luci::CircleTransposeConv *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeBias QuantizeBias::visit node: " << node->name() << std::endl;

  if (auto iwb = IWB(node->outBackprop(), node->filter(), node->bias()))
  {
    auto new_bias = quantized_bias(iwb.input, iwb.weights, iwb.bias);
    node->bias(new_bias);
  }
}

void QuantizeBias::visit(luci::CircleFullyConnected *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeBias visit node: " << node->name() << std::endl;

  if (auto iwb = IWB(node->input(), node->weights(), node->bias()))
  {
    auto new_bias = quantized_bias(iwb.input, iwb.weights, iwb.bias);
    node->bias(new_bias);
  }
}

} // namespace luci
