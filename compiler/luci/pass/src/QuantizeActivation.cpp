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

#include "QuantizeActivation.h"
#include "QuantizationUtils.h"

#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>

#include <algorithm>
#include <cmath>

using namespace luci;

namespace
{

bool has_min_max(const CircleNode *node)
{
  return node->quantparam() && !node->quantparam()->min.empty() && !node->quantparam()->max.empty();
}

} // namespace

// QuantizeActivation
namespace luci
{

void QuantizeActivation::visit(luci::CircleNode *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeActivation visit node: " << node->name() << std::endl;

  // Check if node is fp32
  if (not is_fp32(node))
    return;

  // Check if this is const (const activation is handled by QuantizeConstInputActivation)
  // NOTE QuantizePreChecker guarantees weights/bias are const.
  // Update this code when we accept non-const weights/bias.
  if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    return;

  // Check if this is activation
  // We assume min/max are recorded only for activations
  if (has_min_max(node))
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
      compute_sym_scale(min, max, scaling_factor, nudged_min, nudged_max);
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

} // namespace luci

// QuantizeSpecialActivation
namespace luci
{

void QuantizeSpecialActivation::visit(luci::CircleNode *node)
{
  // Nodes fused with activation functions which need special quantization
  auto fused_act_node = dynamic_cast<CircleNodeMixin<CircleNodeTrait::FusedActFunc> *>(node);
  if (fused_act_node != nullptr && fused_act_node->fusedActivationFunction() == FusedActFunc::TANH)
  {
    auto qparam = make_predefined_qparam(luci::ActivationQType::PreDefinedTanh, output_type,
                                         node->quantparam());
    node->quantparam(std::move(qparam));
  }
}

void QuantizeSpecialActivation::visit(luci::CircleLogistic *node)
{
  auto qparam = make_predefined_qparam(luci::ActivationQType::PreDefinedLogistic, output_type,
                                       node->quantparam());
  node->quantparam(std::move(qparam));
}

void QuantizeSpecialActivation::visit(luci::CircleTanh *node)
{
  auto qparam =
    make_predefined_qparam(luci::ActivationQType::PreDefinedTanh, output_type, node->quantparam());
  node->quantparam(std::move(qparam));
}

void QuantizeSpecialActivation::visit(luci::CircleSoftmax *node)
{
  auto qparam = make_predefined_qparam(luci::ActivationQType::PreDefinedSoftmax, output_type,
                                       node->quantparam());
  node->quantparam(std::move(qparam));
}

void QuantizeSpecialActivation::visit(luci::CircleFloor *node)
{
  assert(activation_qtype(node) == luci::ActivationQType::IntScale);
  set_int_scale(node);
}

void QuantizeSpecialActivation::visit(luci::CircleFloorDiv *node)
{
  assert(activation_qtype(node) == luci::ActivationQType::IntScale);
  set_int_scale(node);
}

void QuantizeSpecialActivation::visit(luci::CircleFloorMod *node)
{
  assert(activation_qtype(node) == luci::ActivationQType::IntScale);
  set_int_scale(node);
}

void QuantizeSpecialActivation::visit(luci::CircleCeil *node)
{
  assert(activation_qtype(node) == luci::ActivationQType::IntScale);
  set_int_scale(node);
}

} // namespace luci

// QuantizeConstInputActivation
namespace luci
{

// Default behavior (NYI)
void QuantizeConstInputActivation::visit(luci::CircleNode *node)
{
  for (uint32_t i = 0; i < node->arity(); i++)
  {
    auto input_node = node->arg(i);
    auto const_node = dynamic_cast<luci::CircleConst *>(input_node);
    if (const_node != nullptr)
    {
      std::string msg = "Unsupported Op for const inputs: " + node->name();
      throw std::runtime_error(msg);
    }
  }
}

// INPUT_NAME is the only activation of NODE
#define QUANTIZE_SINGLE_CONST_INPUT(NODE, INPUT_NAME)           \
  void QuantizeConstInputActivation::visit(NODE *node)          \
  {                                                             \
    auto input = node->INPUT_NAME();                            \
    auto const_node = dynamic_cast<luci::CircleConst *>(input); \
    if (const_node && is_fp32(const_node))                      \
    {                                                           \
      auto new_const = luci::clone(const_node);                 \
      quant_const(new_const, _output_type);                     \
      node->INPUT_NAME(new_const);                              \
    }                                                           \
  }

// INPUT_NAME1 and INPUT_NAME2 are the only activations of NODE
#define QUANTIZE_TWO_CONST_INPUTS(NODE, INPUT_NAME1, INPUT_NAME2) \
  void QuantizeConstInputActivation::visit(NODE *node)            \
  {                                                               \
    auto input1 = node->INPUT_NAME1();                            \
    auto const_node1 = dynamic_cast<luci::CircleConst *>(input1); \
    if (const_node1 && is_fp32(const_node1))                      \
    {                                                             \
      auto new_const1 = luci::clone(const_node1);                 \
      quant_const(new_const1, _output_type);                      \
      node->INPUT_NAME1(new_const1);                              \
    }                                                             \
    auto input2 = node->INPUT_NAME2();                            \
    auto const_node2 = dynamic_cast<luci::CircleConst *>(input2); \
    if (const_node2 && is_fp32(const_node2))                      \
    {                                                             \
      auto new_const2 = luci::clone(const_node2);                 \
      quant_const(new_const2, _output_type);                      \
      node->INPUT_NAME2(new_const2);                              \
    }                                                             \
  }

// Ops that receive a single activation as an input
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleAbs, x)
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleArgMax, input)
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleArgMin, input)
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleBatchToSpaceND, input)
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleDepthToSpace, input)
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleElu, features)
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleExp, x)
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleFloor, x)
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleGather, params)
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleGelu, features)
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
QUANTIZE_SINGLE_CONST_INPUT(luci::CircleSqueeze, input)
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
QUANTIZE_TWO_CONST_INPUTS(luci::CircleFloorMod, x, y)
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
void QuantizeConstInputActivation::visit(luci::CircleAddN *node)
{
  auto arity = node->arity();
  for (uint32_t i = 0; i < arity; i++)
  {
    auto input_node = node->inputs(i);
    auto const_node = dynamic_cast<luci::CircleConst *>(input_node);
    if (const_node && is_fp32(const_node))
    {
      auto new_const = luci::clone(const_node);
      quant_const(new_const, _output_type);
      node->inputs(i, new_const);
    }
  }
}

#undef QUANTIZE_SINGLE_CONST_INPUT
#undef QUANTIZE_TWO_CONST_INPUTS

} // namespace luci
