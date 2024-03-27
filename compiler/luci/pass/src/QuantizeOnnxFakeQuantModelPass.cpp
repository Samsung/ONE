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
#include "luci/Pass/RemoveRedundantTransposePass.h"
#include "QuantizeOnnxQDQPass.h"
#include "QuantizeOnnxDequantizeLinearPass.h"
#include "QuantizeWithPredecessorPass.h"
#include "QuantizeActivation.h"
#include "QuantizationUtils.h"
#include "ProgressReporter.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Log.h>
#include <logo/Phase.h>

namespace
{

// Throw an exception if g is not valid
void verify(loco::Graph *g)
{
#define THROW_UNLESS(cond) \
  if (not(cond))           \
    throw std::runtime_error("Validation failure");

  // Verify the type/granularity of the quantized model
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    // Check fc layers
    if (auto fc = dynamic_cast<luci::CircleFullyConnected *>(node))
    {
      THROW_UNLESS(fc->dtype() == loco::DataType::S16);
      THROW_UNLESS(fc->quantparam() != nullptr);
      auto weights = dynamic_cast<luci::CircleConst *>(fc->weights());
      THROW_UNLESS(weights != nullptr);
      THROW_UNLESS(weights->dtype() == loco::DataType::U4 or
                   weights->dtype() == loco::DataType::S4 or
                   weights->dtype() == loco::DataType::U8);
      THROW_UNLESS(weights->quantparam() != nullptr);
    }

    // TODO Add more checks
  }

#undef THROW_UNLESS
}

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
 * 3. Quantize with predecessors' qparams
 *
 * 4. Quantize remaining fp32 const
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

    QuantizeConstInputActivation qcia(_ctx->default_activation_dtype);
    circle_node->accept(&qcia);
  }

  // Quantize nodes using their predecessors' qparams
  {
    QuantizeWithPredecessorPass pass;
    pass.run(g);
  }

  // Update qparam of output of special Ops
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    if (is_quantized(circle_node))
    {
      QuantizeSpecialActivation qsa(loco::DataType::FLOAT32, circle_node->dtype());
      circle_node->accept(&qsa);
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

  verify(g);

  INFO(l) << "QuantizeOnnxFakeQuantModelPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
