/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "QuantizeWithPredecessorPass.h"

#include "QuantizationUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>

#include <limits>

namespace
{

// Quantize dst node using src node's qparam
// Return true if dst node is quantized with src
// Return false otherwise
bool quantize_with_same_qparam(luci::CircleNode *src, luci::CircleNode *dst)
{
  // src node is not quantized. Skip this case.
  auto src_qparam = src->quantparam();
  if (not src_qparam)
    return false;

  auto dst_qparam = dst->quantparam();
  // dst node is already quantized. Skip this case.
  if (dst_qparam)
    return false;

  luci::copy_quantparam(src, dst);

  dst->dtype(src->dtype());

  return true;
}

//  Visitor to quantize nodes using predecessors qparams
struct QuantizeWithPredecessor final : public luci::CircleNodeMutableVisitor<bool>
{
  QuantizeWithPredecessor() = default;

  bool visit(luci::CircleNode *) { return false; }

  bool visit(luci::CircleReshape *node)
  {
    auto input_node = loco::must_cast<luci::CircleNode *>(node->tensor());
    return quantize_with_same_qparam(input_node, node);
  }

  bool visit(luci::CircleTranspose *node)
  {
    auto input_node = loco::must_cast<luci::CircleNode *>(node->a());
    return quantize_with_same_qparam(input_node, node);
  }

  bool visit(luci::CircleStridedSlice *node)
  {
    auto input_node = loco::must_cast<luci::CircleNode *>(node->input());
    return quantize_with_same_qparam(input_node, node);
  }

  bool visit(luci::CircleSqueeze *node)
  {
    auto input_node = loco::must_cast<luci::CircleNode *>(node->input());
    return quantize_with_same_qparam(input_node, node);
  }

  bool visit(luci::CircleGather *node)
  {
    auto input_node = loco::must_cast<luci::CircleNode *>(node->params());
    return quantize_with_same_qparam(input_node, node);
  }

  bool visit(luci::CircleMul *node)
  {
    // Skip if node is already quantized
    if (luci::is_quantized(node))
      return false;

    auto x = loco::must_cast<luci::CircleNode *>(node->x());
    auto y = loco::must_cast<luci::CircleNode *>(node->y());

    // Only support square for now
    if (x != y)
      return false;

    // Only support S16 for now
    if (x->dtype() != loco::DataType::S16)
      return false;

    const auto input_qparam = x->quantparam();
    if (not input_qparam)
      return false;

    if (input_qparam->scale.size() != 1)
      return false;

    const auto input_scale = input_qparam->scale.at(0);

    const auto s16_max = std::numeric_limits<int16_t>::max();
    // How to determine a new scale of x^2?
    // x's scale would have been determined by its max or min
    //
    // Max value of x^2 = (s * s16_max)^2
    // Min value of x^2 = 0
    // New scale = (s * s16_max)^2 / s16_max = s^2 * s16_max
    //
    // NOTE s16_max = -s16_min (symmetric quantization)
    const auto output_scale = input_scale * input_scale * s16_max;

    auto new_qparam = std::make_unique<luci::CircleQuantParam>();
    {
      new_qparam->scale.push_back(output_scale);
      new_qparam->zerop.push_back(0);
    }

    node->quantparam(std::move(new_qparam));
    node->dtype(x->dtype());

    return true;
  }

  bool visit(luci::CircleNeg *node)
  {
    auto input_node = loco::must_cast<luci::CircleNode *>(node->x());
    // Only support S16 for now
    if (input_node->dtype() != loco::DataType::S16)
      return false;

    return quantize_with_same_qparam(input_node, node);
  }

  bool visit(luci::CircleConcatenation *node)
  {
    const auto num_inputs = node->numValues();

    for (uint32_t i = 0; i < num_inputs; i++)
    {
      auto input = loco::must_cast<luci::CircleNode *>(node->values(i));
      // Only support S16 for now
      if (input->dtype() != loco::DataType::S16)
        return false;

      if (input->quantparam() == nullptr)
        return false;

      if (input->quantparam()->scale.size() != 1)
        return false;
    }

    luci::CircleNode *max_scale_node = nullptr;
    float max_scale = 0.0;
    for (uint32_t i = 0; i < num_inputs; i++)
    {
      auto input = loco::must_cast<luci::CircleNode *>(node->values(i));
      auto qparam = input->quantparam();
      auto scale = qparam->scale.at(0);
      if (max_scale < scale)
      {
        max_scale = scale;
        max_scale_node = input;
      }
    }

    assert(max_scale_node);

    return quantize_with_same_qparam(max_scale_node, node);
  }
};

} // namespace

namespace luci
{

bool QuantizeWithPredecessorPass::run(loco::Graph *g)
{
  bool changed = false;

  LOGGER(l);

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    INFO(l) << "QuantizeWithPredecessorPass visit node: " << circle_node->name() << std::endl;

    QuantizeWithPredecessor qwp;
    if (circle_node->accept(&qwp))
    {
      changed = true;
    }
  }

  return changed;
}

} // namespace luci
