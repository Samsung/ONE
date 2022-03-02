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
#include "QuantizeWeights.h"
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

namespace
{

using namespace luci;
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

    // Check if this is already quantized
    if (is_quantized(node))
      return;

    // Check if this is bias (bias is quantized later)
    auto iwo = get_input_weight_output_of_bias(node);
    if (iwo.size() > 0)
      return;

    // Check if this is bool type (bool type is not quantized)
    if (node->dtype() == loco::DataType::BOOL)
      return;

    // Check if this is activation
    // We assume min/max are recorded only for activations
    if (has_min_max(node) && !is_weights(node))
    {
      // Quantize using recorded min/max
      auto quantparam = node->quantparam();
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
        node->dtype(loco::DataType::U8);
      }
      else
      {
        compute_sym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
        node->dtype(loco::DataType::S16);
      }

      node->quantparam()->scale.push_back(scaling_factor);
      node->quantparam()->zerop.push_back(zp);
    }
    // Fix special attributes
    if (node->opcode() == luci::CircleOpcode::CAST)
    {
      auto *cast = loco::must_cast<luci::CircleCast *>(node);
      auto *cast_input = loco::must_cast<luci::CircleNode *>(cast->x());

      // make sure that cast_input is already quantized
      assert(cast_input->dtype() != loco::DataType::FLOAT32);
      cast->in_data_type(cast_input->dtype());
      cast->out_data_type(cast->dtype());
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

private:
  // Return a quantized bias node
  CircleConst *quantized_bias(CircleNode *input, const CircleNode *weight, CircleNode *bias)
  {
    auto const_bias = loco::must_cast<luci::CircleConst *>(bias);
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

  void visit(luci::CircleConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeBias visit node: " << node->name() << std::endl;

    auto bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (not bias)
      return;

    if (is_quantized(bias))
      return;

    auto i = loco::must_cast<luci::CircleNode *>(node->input());
    auto w = loco::must_cast<luci::CircleNode *>(node->filter());
    auto new_bias = quantized_bias(i, w, bias);
    node->bias(new_bias);
  }

  void visit(luci::CircleDepthwiseConv2D *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeBias visit node: " << node->name() << std::endl;

    auto bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (not bias)
      return;

    if (is_quantized(bias))
      return;

    auto i = loco::must_cast<luci::CircleNode *>(node->input());
    auto w = loco::must_cast<luci::CircleNode *>(node->filter());
    auto new_bias = quantized_bias(i, w, bias);
    node->bias(new_bias);
  }

  void visit(luci::CircleTransposeConv *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeBias visit node: " << node->name() << std::endl;

    auto bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (not bias)
      return;

    if (is_quantized(bias))
      return;

    auto i = loco::must_cast<luci::CircleNode *>(node->outBackprop());
    auto w = loco::must_cast<luci::CircleNode *>(node->filter());
    auto new_bias = quantized_bias(i, w, bias);
    node->bias(new_bias);
  }

  void visit(luci::CircleFullyConnected *node)
  {
    LOGGER(l);
    INFO(l) << "QuantizeBias visit node: " << node->name() << std::endl;

    auto bias = dynamic_cast<luci::CircleConst *>(node->bias());
    if (not bias)
      return;

    if (is_quantized(bias))
      return;

    auto i = loco::must_cast<luci::CircleNode *>(node->input());
    auto w = loco::must_cast<luci::CircleNode *>(node->weights());
    auto new_bias = quantized_bias(i, w, bias);
    node->bias(new_bias);
  }

  void visit(luci::CircleNode *) {}
};

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
  SKIP(luci::CirclePadV2)
  SKIP(luci::CirclePack)
  SKIP(luci::CircleOneHot)

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

#undef SKIP
#undef QUANTIZE_SINGLE_CONST_INPUT
#undef QUANTIZE_TWO_CONST_INPUTS
};

} // namespace

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
 *   - Quantize const inputs (QuantizeConstInputActivation)
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
