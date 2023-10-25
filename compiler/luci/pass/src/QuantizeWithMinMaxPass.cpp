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
#include "luci/Pass/RemoveRedundantQuantizePass.h"
#include "QuantizeActivation.h"
#include "QuantizeWeights.h"
#include "QuantizeBias.h"
#include "QuantizationUtils.h"
#include "ProgressReporter.h"
#include "helpers/LayerInfoMap.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Log.h>
#include <logo/Phase.h>

#include <iostream>
#include <cmath>

namespace
{

using namespace luci;

bool use_predefined_values(ActivationQType qtype)
{
  switch (qtype)
  {
    case ActivationQType::PreDefinedLogistic:
    case ActivationQType::PreDefinedTanh:
    case ActivationQType::PreDefinedSoftmax:
      return true;
    default:
      // This ensures this switch-statement handles all ActivationQTypes
      assert(qtype == ActivationQType::IntScale or qtype == ActivationQType::MinMax);
      break;
  }

  return false;
}

// Create a Quantize Op whose
// dtype is out_type
// shape is the same with node
// qparam is computed according to node's qtype
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
  assert(qparam); // FIX_CALLER_UNLESS

  auto qtype = luci::activation_qtype(node);
  if (use_predefined_values(qtype))
  {
    quantize->quantparam(luci::make_predefined_qparam(qtype, out_type, node->quantparam()));
    return quantize;
  }

  assert(qtype == ActivationQType::MinMax or qtype == ActivationQType::IntScale);

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
    compute_sym_scale(min, max, scaling_factor, nudged_min, nudged_max);
  }

  auto quantparam = std::make_unique<CircleQuantParam>();
  quantparam->scale.push_back(scaling_factor);
  quantparam->zerop.push_back(zp);
  // Save original min/max (not nudged_min/max). Nudged min/max
  // is different from the real min/max values, causing wrong
  // qparam when quantization dtype is changed.
  quantparam->min.push_back(min);
  quantparam->max.push_back(max);

  quantize->quantparam(std::move(quantparam));

  if (qtype == ActivationQType::IntScale)
    set_int_scale(quantize);

  return quantize;
}

// Create Dequantize Op whose shape is the same with node
luci::CircleDequantize *create_dequantize(luci::CircleNode *node)
{
  auto dequantize = node->graph()->nodes()->create<luci::CircleDequantize>();
  dequantize->name(node->name() + "_Dequantize");
  dequantize->dtype(loco::DataType::FLOAT32);
  dequantize->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); i++)
    dequantize->dim(i).set(node->dim(i).value());

  dequantize->shape_status(luci::ShapeStatus::VALID);

  luci::add_origin(dequantize, luci::get_origin(node));

  return dequantize;
}

} // namespace

namespace luci
{

namespace
{

/**
 * Insert Quantize operator for mixed-precision quantization
 * 1. Before input feature map (only for non-const)
 * 2. After output feature map
 *
 * For example, if default_dtype = U8 and op_dtype = S16,
 * 1. Quantize (U8->S16) is inserted before ifm
 * 2. Quantize (S16->U8) is inserted after ofm
 *
 * Why not insert Quantize Op for const ifm?
 * We quantize const tensor at once to preserve precision.
 * For example, if default dtype = U8, op_dtype = S16, and op is CONV2D,
 * We directly quantize weights to 16 bits, not 8->16 bits.
 */
struct InsertQuantizeOp final : public luci::CircleNodeMutableVisitor<void>
{
  InsertQuantizeOp(loco::DataType default_dtype, loco::DataType op_dtype)
    : _default_dtype(default_dtype), _op_dtype(op_dtype)
  {
    assert(default_dtype != op_dtype); // FIX_CALLER_UNLESS
  }

private:
  loco::DataType _default_dtype;
  loco::DataType _op_dtype;

private:
  luci::CircleQuantize *create_in_quantize(loco::Node *in, loco::Node *origin)
  {
    auto input = loco::must_cast<luci::CircleNode *>(in);
    if (input->opcode() == luci::CircleOpcode::CIRCLECONST)
      return nullptr;

    // input is not quantizable (ex: index)
    if (input->quantparam() == nullptr)
      return nullptr;

    auto input_quant = create_quantize_op(input, _op_dtype);
    input_quant->input(input);
    auto origin_node = loco::must_cast<luci::CircleNode *>(origin);
    luci::add_origin(input_quant, luci::get_origin(origin_node));
    return input_quant;
  }

  void insert_out_quantize(loco::Node *node)
  {
    auto output = loco::must_cast<luci::CircleNode *>(node);
    assert(output->opcode() != luci::CircleOpcode::CIRCLECONST); // FIX_CALLER_UNLESS

    // output is not quantizable (ex: index)
    if (output->quantparam() == nullptr)
      return;

    auto output_quant = create_quantize_op(output, _default_dtype);

    luci::add_origin(output_quant, luci::get_origin(output));
    loco::replace(node).with(output_quant);
    output_quant->input(node);
  }

// INPUT_NAME is the only activation of NODE
#define INSERT_QUANTIZE_TO_UNARY_OP(NODE, INPUT_NAME)                    \
  void visit(NODE *node)                                                 \
  {                                                                      \
    if (auto input_quant = create_in_quantize(node->INPUT_NAME(), node)) \
      node->INPUT_NAME(input_quant);                                     \
                                                                         \
    insert_out_quantize(node);                                           \
  }

// INPUT_NAME is the only activation of NODE
#define INSERT_QUANTIZE_TO_UNARY_MULTI_OUTPUT_OP(NODE, INPUT_NAME, OUT_NAME) \
  void visit(NODE *node)                                                     \
  {                                                                          \
    if (auto input_quant = create_in_quantize(node->INPUT_NAME(), node))     \
      node->INPUT_NAME(input_quant);                                         \
                                                                             \
    auto out_nodes = loco::succs(node);                                      \
    for (auto out_node : out_nodes)                                          \
    {                                                                        \
      auto out_circle = loco::must_cast<OUT_NAME *>(out_node);               \
      insert_out_quantize(out_circle);                                       \
    }                                                                        \
  }

// INPUT_NAME1 and INPUT_NAME2 are the only activations of NODE
#define INSERT_QUANTIZE_TO_BINARY_OP(NODE, INPUT_NAME1, INPUT_NAME2)       \
  void visit(NODE *node)                                                   \
  {                                                                        \
    if (auto input1_quant = create_in_quantize(node->INPUT_NAME1(), node)) \
      node->INPUT_NAME1(input1_quant);                                     \
                                                                           \
    if (auto input2_quant = create_in_quantize(node->INPUT_NAME2(), node)) \
      node->INPUT_NAME2(input2_quant);                                     \
                                                                           \
    insert_out_quantize(node);                                             \
  }

  // Default behavior (NYI)
  void visit(luci::CircleNode *node)
  {
    throw std::runtime_error("Unsupported Op for mixed-precision quantization. Layer name: " +
                             node->name());
  }

  // Skip output layer
  void visit(luci::CircleOutput *) {}
  void visit(luci::CircleSplitVOut *) {}
  void visit(luci::CircleSplitOut *) {}
  void visit(luci::CircleTopKV2Out *) {}
  void visit(luci::CircleUniqueOut *) {}
  void visit(luci::CircleUnpackOut *) {}

  // Ops that receive a single activation as an input
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleAbs, x)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleAveragePool2D, value)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleBatchToSpaceND, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleConv2D, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleDepthToSpace, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleDepthwiseConv2D, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleElu, features)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleExp, x)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleFloor, x)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleFullyConnected, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleGather, params)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleGelu, features)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleInstanceNorm, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleLeakyRelu, features)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleLocalResponseNormalization, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleLogistic, x)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleMaxPool2D, value)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleMean, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleMirrorPad, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleNeg, x)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CirclePad, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CirclePadV2, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CirclePRelu, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleReduceProd, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleReduceMax, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleReduceMin, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleRelu, features)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleRelu6, features)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleReshape, tensor)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleResizeBilinear, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleResizeNearestNeighbor, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleReverseSequence, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleRsqrt, x)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleSlice, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleSoftmax, logits)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleSpaceToBatchND, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleSpaceToDepth, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleSqueeze, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleSqrt, x)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleStridedSlice, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleSum, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleTanh, x)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleTile, input)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleTranspose, a)
  INSERT_QUANTIZE_TO_UNARY_OP(luci::CircleTransposeConv, outBackprop)

  // Ops that receive two activations as inputs
  INSERT_QUANTIZE_TO_BINARY_OP(luci::CircleAdd, x, y)
  INSERT_QUANTIZE_TO_BINARY_OP(luci::CircleBatchMatMul, x, y)
  INSERT_QUANTIZE_TO_BINARY_OP(luci::CircleDiv, x, y)
  INSERT_QUANTIZE_TO_BINARY_OP(luci::CircleFloorDiv, x, y)
  INSERT_QUANTIZE_TO_BINARY_OP(luci::CircleMaximum, x, y)
  INSERT_QUANTIZE_TO_BINARY_OP(luci::CircleMinimum, x, y)
  INSERT_QUANTIZE_TO_BINARY_OP(luci::CircleMul, x, y)
  INSERT_QUANTIZE_TO_BINARY_OP(luci::CircleOneHot, on_value, off_value)
  INSERT_QUANTIZE_TO_BINARY_OP(luci::CirclePow, x, y)
  INSERT_QUANTIZE_TO_BINARY_OP(luci::CircleSub, x, y)

  // Multiple-output ops that receive one activation as inputs
  INSERT_QUANTIZE_TO_UNARY_MULTI_OUTPUT_OP(luci::CircleSplit, input, luci::CircleSplitOut)
  INSERT_QUANTIZE_TO_UNARY_MULTI_OUTPUT_OP(luci::CircleSplitV, input, luci::CircleSplitVOut)
  INSERT_QUANTIZE_TO_UNARY_MULTI_OUTPUT_OP(luci::CircleTopKV2, input, luci::CircleTopKV2Out)
  INSERT_QUANTIZE_TO_UNARY_MULTI_OUTPUT_OP(luci::CircleUnique, input, luci::CircleUniqueOut)
  INSERT_QUANTIZE_TO_UNARY_MULTI_OUTPUT_OP(luci::CircleUnpack, value, luci::CircleUnpackOut)

  // AddN has arbitrary number of inputs
  void visit(luci::CircleAddN *node)
  {
    auto arity = node->arity();
    for (uint32_t i = 0; i < arity; i++)
    {
      if (auto input_quant = create_in_quantize(node->inputs(i), node))
        node->inputs(i, input_quant);
    }

    insert_out_quantize(node);
  }

  // Concat has arbitrary number of inputs
  void visit(luci::CircleConcatenation *node)
  {
    auto arity = node->arity();
    for (uint32_t i = 0; i < arity; i++)
    {
      if (auto input_quant = create_in_quantize(node->values(i), node))
        node->values(i, input_quant);
    }

    insert_out_quantize(node);
  }

  // Pack has arbitrary number of inputs
  void visit(luci::CirclePack *node)
  {
    auto arity = node->arity();
    for (uint32_t i = 0; i < arity; i++)
    {
      if (auto input_quant = create_in_quantize(node->values(i), node))
        node->values(i, input_quant);
    }

    insert_out_quantize(node);
  }

#undef INSERT_QUANTIZE_TO_UNARY_OP
#undef INSERT_QUANTIZE_TO_BINARY_OP
#undef INSERT_QUANTIZE_TO_UNARY_MULTI_OUTPUT_OP
};

} // namespace

void QuantizeWithMinMaxPass::set_input_type(loco::Graph *g) const
{
  auto inputs = g->inputs();

  assert(inputs);                                     // FIX_CALLER_UNLESS
  assert(inputs->size() == _ctx->input_types.size()); // FIX_CALLER_UNLESS

  // NOTE loco::input_nodes returns input nodes following the order of InputIndex
  auto input_nodes = loco::input_nodes(g);
  for (uint32_t i = 0; i < input_nodes.size(); i++)
  {
    auto input = loco::must_cast<luci::CircleInput *>(input_nodes[i]);
    assert(i == input->index()); // Fix input_type logic

    const auto user_given_dtype = _ctx->input_types[i];

    if (input->dtype() == user_given_dtype)
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

    // Update qparam of input
    // This step is skipped if input_type is float32
    if (user_given_dtype != loco::DataType::FLOAT32)
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

      if (user_given_dtype == loco::DataType::U8)
      {
        compute_asym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
      }
      else
      {
        assert(user_given_dtype == loco::DataType::S16);
        compute_sym_scale(min, max, scaling_factor, nudged_min, nudged_max);
      }
      input->quantparam()->scale[0] = scaling_factor;
      input->quantparam()->zerop[0] = zp;
    }

    // Update dtype of input
    input->dtype(user_given_dtype);

    auto graph_input = inputs->at(input->index());
    graph_input->dtype(user_given_dtype);
  }
}

void QuantizeWithMinMaxPass::set_output_type(loco::Graph *g) const
{
  auto outputs = g->outputs();
  assert(outputs);                                      // FIX_CALLER_UNLESS
  assert(outputs->size() == _ctx->output_types.size()); // Fix CircleQuantizer unless

  // NOTE loco::output_nodes returns output nodes following the order of OutputIndex
  auto output_nodes = loco::output_nodes(g);
  for (uint32_t i = 0; i < output_nodes.size(); i++)
  {
    auto output = loco::must_cast<luci::CircleOutput *>(output_nodes[i]);
    assert(i == output->index()); // Fix output_type logic

    const auto user_given_dtype = _ctx->output_types[i];

    if (output->dtype() == user_given_dtype)
      continue;

    // Bool type is not quantizable
    if (output->dtype() == loco::DataType::BOOL)
      continue;

    auto from = loco::must_cast<luci::CircleNode *>(output->from());

    // The last Op is not quantizable (ex: ArgMax)
    if (not from->quantparam())
      continue;

    // Insert Dequantize Op for float32 output_type
    if (user_given_dtype == loco::DataType::FLOAT32)
    {
      auto dequant_op = create_dequantize(from);
      dequant_op->input(from);
      output->from(dequant_op);
    }
    else
    {
      // Insert Quantize Op for non-float32 output_type
      auto quant_op = create_quantize_op(from, user_given_dtype);
      quant_op->input(from);
      output->from(quant_op);

      // TODO Set a proper origin (Quantize should have its own Origin)
      luci::add_origin(quant_op, luci::get_origin(from));
    }

    // Update dtype of output
    output->dtype(user_given_dtype);

    auto graph_output = outputs->at(output->index());
    graph_output->dtype(user_given_dtype);
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
 * - Therefore, activation and weights should be quantized earlier than bias
 *
 * Overall Quantization Steps
 * 1. Quantize Activation
 *   - Quantize using recorded min/max (QuantizeActivation)
 *   - Insert Quantize Ops for mixed-precision quantization (InsertQuantizeOp)
 *   - Remove redundant Quantize Ops (RemoveRedundantQuantizePass)
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

  // Quantize activation
  // Why all_nodes?
  // Models can have inactive (unused) inputs.
  // We do not reject such models, but quantize them too
  for (auto node : loco::all_nodes(g))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    QuantizeActivation qa(_ctx->input_model_dtype, quantize_dtype(circle_node));
    circle_node->accept(&qa);
  }

  // Insert Quantize Op
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    auto op_dtype = quantize_dtype(circle_node);
    if (op_dtype != _ctx->output_model_dtype)
    {
      InsertQuantizeOp iqo(_ctx->output_model_dtype, op_dtype);
      circle_node->accept(&iqo);
    }
  }

  // Remove redundant Quantize Op
  {
    logo::Phase phase;

    phase.emplace_back(std::make_unique<luci::RemoveRedundantQuantizePass>());

    ProgressReporter prog(g, logo::PhaseStrategy::Saturate);
    logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{g};
    phase_runner.attach(&prog);
    phase_runner.run(phase);
  }

  // Backward propagation of activation qparam
  {
    PropagateQParamBackwardPass pqbp(_ctx->output_model_dtype);
    pqbp.run(g);
  }

  // Quantize const input activation
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    QuantizeConstInputActivation qcia(quantize_dtype(circle_node));
    circle_node->accept(&qcia);
  }

  // Update qparam of output of special Ops
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    // At this point, all activations have to be quantized.
    // Un-quantized nodes are not the quantization target (ex: int32 tensor),
    // so we skip them
    if (circle_node->quantparam() == nullptr)
      continue;

    QuantizeSpecialActivation qsa(_ctx->input_model_dtype, quantize_dtype(circle_node));
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
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    QuantizeWeights qw(_ctx->input_model_dtype, quantize_dtype(circle_node),
                       quantize_granularity(circle_node));
    circle_node->accept(&qw);
  }

  // Quantize bias
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    QuantizeBias qb(_ctx->input_model_dtype, quantize_dtype(circle_node),
                    quantize_granularity(circle_node));
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

  // Remove redundant Quantize Op
  {
    logo::Phase phase;

    phase.emplace_back(std::make_unique<luci::RemoveRedundantQuantizePass>());

    ProgressReporter prog(g, logo::PhaseStrategy::Saturate);
    logo::PhaseRunner<logo::PhaseStrategy::Saturate> phase_runner{g};
    phase_runner.attach(&prog);
    phase_runner.run(phase);
  }

  if (not _ctx->save_min_max)
  {
    // Remove min/max values
    for (auto node : loco::all_nodes(g))
    {
      auto circle_node = loco::must_cast<luci::CircleNode *>(node);
      if (auto qparam = circle_node->quantparam())
      {
        warn_accuracy_with_range(circle_node);
        qparam->min.clear();
        qparam->max.clear();
      }
    }
  }

  INFO(l) << "QuantizeWithMinMaxPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
