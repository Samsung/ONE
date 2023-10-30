/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseUnrolledGRUAsCustomGRU.h"

#include <luci/IR/CircleNode.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

/**
 *  Fuse Transpose with Mean if possible
 *
 *  BEFORE
 *                  |
 *          [CircleTranspose, perm<0, 2, 3, 1>]
 *                  |
 *          [CircleMean, axis<3>]
 *                  |
 *
 *  AFTER
 *                  |                            |
 *          [CircleMean, axis<1>]       [CircleTranspose, perm<0, 2, 3, 1>]
 *                  |                            |
 *                                      [CircleMean, axis<3>]
 *
 */

/**
 * @brief Create a const for fused reduction indices
 */
luci::CircleConst *create_fused_indices(luci::CircleConst *rindices,
                                        const std::vector<uint32_t> &fused_rindices)
{
  assert(rindices != nullptr); // FIX_CALLER_UNLESS

  if (rindices->dtype() != loco::DataType::S32)
    return nullptr;

  assert(fused_rindices.size() == rindices->size<loco::DataType::S32>());

  auto fused_rindices_const = luci::clone(rindices);
  auto name = rindices->name();
  assert(name.length() > 0); // FIX_CALLER_UNLESS
  fused_rindices_const->name(name + "_fused");

  for (uint32_t i = 0; i < fused_rindices.size(); ++i)
  {
    fused_rindices_const->at<loco::DataType::S32>(i) = fused_rindices.at(i);
  }

  return fused_rindices_const;
}

bool const_has_value_s32(const luci::CircleConst *circle_const, int32_t value)
{
  if (circle_const->dtype() != loco::DataType::S32)
    return false;

  uint32_t size = circle_const->size<loco::DataType::S32>();
  for (uint32_t i = 0; i < size; ++i)
  {
    if (circle_const->at<loco::DataType::S32>(i) == value)
      return true;
  }

  return false;
}

bool create_custom_op(luci::CircleWhile *while_node)
{
  auto input_node = dynamic_cast<luci::CircleNode *>(while_node->input(4));
  auto input_state_node = dynamic_cast<luci::CircleNode *>(while_node->input(3));
  loco::Node *weight_ih;
  loco::Node *bias_ih;

  loco::Node *weight_hh;
  loco::Node *bias_hh;

  auto input_size = input_node->dim(input_node->rank() - 1).value();
  auto hidden_size = input_state_node->dim(input_state_node->rank() - 1).value();

  auto body_graph = while_node->body_graph();
  for (auto node : loco::active_nodes(loco::output_nodes(body_graph)))
  {
    auto fc = dynamic_cast<luci::CircleFullyConnected *>(node);
    if (not fc)
      continue;
    if (fc->dim(fc->rank() - 1) == input_size)
    {
      weight_ih = fc->weights();
      bias_ih = fc->bias();
    }
    if (fc->dim(fc->rank() - 1) == hidden_size)
    {
      weight_hh = fc->weights();
      bias_hh = fc->bias();
    }
  }

  // Create and configure new CircleMean operation.
  auto fused_gru = while_node->graph()->nodes()->create<luci::CircleCustom>(6, 2);
  fused_gru->custom_code("custom_gru");
  fused_gru->inputs(0, input_node);

  return true;
}

} // namespace

namespace luci
{

bool FuseUnrolledGRUAsCustomGRUPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto mean = dynamic_cast<luci::CircleWhile *>(node);
    if (not mean)
      continue;

    if (create_custom_op(mean))
      changed = true;
  }

  return changed;
}

} // namespace luci
