/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/QuantizeOnnxFakeQuantModelPass.h"
#include "luci/Pass/FoldSqueezePass.h"
#include "QuantizeOnnxQDQPass.h"
#include "QuantizeOnnxDequantizeLinearPass.h"
#include "QuantizeWithPredecessorPass.h"
#include "QuantizeActivation.h"
#include "QuantizationUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Log.h>

#include <limits> // std::numeric_limits

using namespace luci;

namespace
{

// Update u8 node to i16
// dtype and qparam are updated
void update_u8_to_i16(luci::CircleNode *node)
{
  assert(node->dtype() == loco::DataType::U8); // FIX_CALLER_UNLESS

  node->dtype(loco::DataType::S16);

  auto qparam = node->quantparam();
  assert(qparam);
  assert(qparam->scale.size() == 1);
  assert(qparam->zerop.size() == 1);

  auto u8_scale = qparam->scale[0];
  auto u8_zerop = qparam->zerop[0];

  auto min = u8_scale * (-u8_zerop);
  auto max = u8_scale * (255 - u8_zerop);

  float s16_scale{0};
  int64_t s16_zerop{0};
  float nudged_min{0};
  float nudged_max{0};

  compute_sym_scale(min, max, s16_scale, nudged_min, nudged_max);

  auto quantparam = std::make_unique<CircleQuantParam>();
  quantparam->scale.push_back(s16_scale);
  quantparam->zerop.push_back(s16_zerop);

  node->quantparam(std::move(quantparam));
}

void update_i16_to_u8(luci::CircleNode *node)
{
  assert(node->dtype() == loco::DataType::S16); // FIX_CALLER_UNLESS

  node->dtype(loco::DataType::U8);

  auto qparam = node->quantparam();
  assert(qparam);
  assert(qparam->scale.size() == 1);
  assert(qparam->zerop.size() == 1);

  auto s16_scale = qparam->scale[0];
  assert(qparam->zerop[0] == 0);

  auto max = s16_scale * std::numeric_limits<int16_t>::max();
  auto min = -max;

  float u8_scale{0};
  int64_t u8_zerop{0};
  float nudged_min{0};
  float nudged_max{0};

  compute_asym_scale_zp(min, max, u8_scale, u8_zerop, nudged_min, nudged_max);

  auto quantparam = std::make_unique<CircleQuantParam>();
  quantparam->scale.push_back(u8_scale);
  quantparam->zerop.push_back(u8_zerop);

  node->quantparam(std::move(quantparam));
}

// Create a Quantize Op whose
// dtype, shape, and qparam are the same with node's
luci::CircleQuantize *create_quantize_op(luci::CircleNode *node)
{
  auto quantize = node->graph()->nodes()->create<CircleQuantize>();
  quantize->name(node->name() + "_Quantize");
  quantize->dtype(node->dtype());
  quantize->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); i++)
    quantize->dim(i).set(node->dim(i).value());

  quantize->shape_status(luci::ShapeStatus::VALID);

  assert(node->quantparam()); // FIX_CALLER_UNLESS
  copy_quantparam(node, quantize);

  luci::add_origin(quantize, luci::get_origin(node));

  return quantize;
}

// Insert Quantize Op if input's dtype mismatches with output's dtype
// NOTE Mismatch is determined by CircleTypeInferenceRule
// NOTE Find a better approach to break coupling with CircleTypeInferenceRule
struct InsertQuantizeOpOnDtypeMismatch final : public luci::CircleNodeMutableVisitor<void>
{
  InsertQuantizeOpOnDtypeMismatch() = default;

private:
  void visit(luci::CircleNode *) {}

  void visit(luci::CircleFullyConnected *node)
  {
    auto input = loco::must_cast<luci::CircleNode *>(node->input());

    // Input dtype == Output dtype. No problem
    if (input->dtype() == node->dtype())
      return;

    // Skip if node has bias
    if (dynamic_cast<luci::CircleOutputExclude *>(node->bias()) == nullptr)
      return;

    if (node->fusedActivationFunction() != luci::FusedActFunc::NONE)
      return;

    // Only cares quantized case
    if (not is_quantized(input))
      return;

    if (not is_quantized(node))
      return;

    // Let's support limited case
    // TODO Extend this to another dtype
    if (input->dtype() != loco::DataType::U8)
      return;

    if (node->dtype() != loco::DataType::S16)
      return;

    // Create Quantize Op
    auto quant_op = create_quantize_op(node);

    // Insert Quantize Op after node
    loco::replace(node).with(quant_op);
    quant_op->input(node);

    // Update node's dtype and qparam from i16 to u8
    update_i16_to_u8(node);
  }

  void visit(luci::CircleMul *node)
  {
    auto x = loco::must_cast<luci::CircleNode *>(node->x());
    auto y = loco::must_cast<luci::CircleNode *>(node->y());

    assert(x->dtype() == y->dtype()); // FIX_CALLER_UNLESS

    // Ignore invalid dtype
    if (x->dtype() != y->dtype())
      return;

    if (node->fusedActivationFunction() != luci::FusedActFunc::NONE)
      return;

    // Input dtype == Output dtype. No problem
    if (x->dtype() == node->dtype())
      return;

    // Only cares quantized case
    if (not is_quantized(x))
      return;

    if (not is_quantized(y))
      return;

    if (not is_quantized(node))
      return;

    // Let's support limited case
    // TODO Extend this to another dtype
    if (x->dtype() != loco::DataType::S16)
      return;

    if (node->dtype() != loco::DataType::U8)
      return;

    // Create Quantize Op
    auto quant_op = create_quantize_op(node);

    // Insert Quantize Op after node
    loco::replace(node).with(quant_op);
    quant_op->input(node);

    // Update node's dtype and qparam from u8 to i16
    update_u8_to_i16(node);
  }

  void visit(luci::CircleBatchMatMul *node)
  {
    auto x = loco::must_cast<luci::CircleNode *>(node->x());
    auto y = loco::must_cast<luci::CircleNode *>(node->y());

    assert(x->dtype() == y->dtype()); // FIX_CALLER_UNLESS

    // Ignore invalid dtype
    if (x->dtype() != y->dtype())
      return;

    if (node->adj_x() or node->adj_y())
      return;

    // Input dtype == Output dtype. No problem
    if (x->dtype() == node->dtype())
      return;

    // Only cares quantized case
    if (not is_quantized(x))
      return;

    if (not is_quantized(y))
      return;

    if (not is_quantized(node))
      return;

    // Let's support limited case
    // TODO Extend this to another dtype
    if (x->dtype() != loco::DataType::S16)
      return;

    if (node->dtype() != loco::DataType::U8)
      return;

    // Create Quantize Op
    auto quant_op = create_quantize_op(node);

    // Insert Quantize Op after node
    loco::replace(node).with(quant_op);
    quant_op->input(node);

    // Update node's dtype and qparam from i16 to u8
    // NOTE This would severely degrade accuracy. It is
    // important to mitigate this accuracy drop in backend.
    update_u8_to_i16(node);
  }
};

} // namespace

namespace luci
{

/**
 * How QuantizeOnnxFakeQuantModel works?
 *
 * 1. Activation is quantized as below.
 *
 * Before
 *
 * [node(fp32)] -> [OnnxQuantizeLinear] -> [OnnxDequantizeLinear]
 *
 * After
 *
 * [node(q)]
 *
 *
 * 2. Weight(constant) are quantized as below.
 *
 * Before
 *
 * [Const(q w/o qparam)] -> [OnnxDequantizeLinear]
 *
 * After
 *
 * [Const(q)]
 *
 * 3. Quantize constant activations
 *
 * 4. Quantize with predecessors' qparams
 *
 * 5. Update qparams of special operators
 */
bool QuantizeOnnxFakeQuantModelPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "QuantizeOnnxFakeQuantModelPass Start" << std::endl;

  // Quantize Onnx QuantizeLinear-DequantizeLinear pattern
  {
    QuantizeOnnxQDQPass pass;
    pass.run(g);
  }

  // Quantize Onnx const-DequantizeLinear pattern
  {
    QuantizeOnnxDequantizeLinearPass pass;
    pass.run(g);
  }

  // Quantize const input activation
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    QuantizeConstInputActivation visitor(_ctx->default_activation_dtype);
    circle_node->accept(&visitor);
  }

  // Quantize nodes using their predecessors' qparams
  {
    QuantizeWithPredecessorPass pass;
    pass.run(g);
  }

  // Insert QuantizeOp if input/output dtype does not match
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    InsertQuantizeOpOnDtypeMismatch visitor;
    circle_node->accept(&visitor);
  }

  // Update qparam of output of special Ops
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    if (is_quantized(circle_node))
    {
      QuantizeSpecialActivation visitor(circle_node->dtype());
      circle_node->accept(&visitor);
    }
  }

  // Update output dtype
  auto graph_outputs = g->outputs();
  for (auto node : loco::output_nodes(g))
  {
    auto circle_node = loco::must_cast<luci::CircleOutput *>(node);
    auto from = loco::must_cast<luci::CircleNode *>(circle_node->from());
    circle_node->dtype(from->dtype());

    auto graph_output = graph_outputs->at(circle_node->index());
    graph_output->dtype(circle_node->dtype());
  }

  INFO(l) << "QuantizeOnnxFakeQuantModelPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
