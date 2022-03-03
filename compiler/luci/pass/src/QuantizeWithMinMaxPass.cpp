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
#include "QuantizeActivation.h"
#include "QuantizeWeights.h"
#include "QuantizeBias.h"
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
  // Save original min/max (not nudged_min/max). Nudged min/max
  // is different from the real min/max values, causing wrong
  // qparam when quantization dtype is changed.
  quantparam->min.push_back(min);
  quantparam->max.push_back(max);

  quantize->quantparam(std::move(quantparam));

  return quantize;
}

} // namespace

namespace luci
{

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
