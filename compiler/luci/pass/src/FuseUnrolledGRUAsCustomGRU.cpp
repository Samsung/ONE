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
#include "luci/Service/CircleNodeClone.h"

#include <luci/IR/CircleNode.h>
#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

template <loco::DataType T>
void copy_values(const luci::CircleConst *node, luci::CircleConst *cloned)
{
  assert(T == node->dtype());
  assert(T == cloned->dtype());

  const auto size = node->size<T>();
  cloned->size<T>(size);
  for (uint32_t i = 0; i < size; i++)
    cloned->at<T>(i) = node->at<T>(i);
}

luci::CircleConst *clone_circleconst(luci::CircleConst *node, loco::Graph *graph)
{
  auto cloned = graph->nodes()->create<luci::CircleConst>();

  if (cloned != nullptr)
  {
    // dtype/shape
    cloned->dtype(node->dtype());
    cloned->rank(node->rank());

    // values
    switch (node->dtype())
    {
      case loco::DataType::FLOAT32:
        copy_values<loco::DataType::FLOAT32>(node, cloned);
        break;

      case loco::DataType::U8:
        copy_values<loco::DataType::U8>(node, cloned);
        break;

      case loco::DataType::S8:
        copy_values<loco::DataType::S8>(node, cloned);
        break;

      case loco::DataType::S16:
        copy_values<loco::DataType::S16>(node, cloned);
        break;

      case loco::DataType::S32:
        copy_values<loco::DataType::S32>(node, cloned);
        break;

      case loco::DataType::S64:
        copy_values<loco::DataType::S64>(node, cloned);
        break;

      case loco::DataType::BOOL:
        copy_values<loco::DataType::BOOL>(node, cloned);
        break;

      default:
        assert(false);
    }
  }

  return cloned;
}

} // namespace

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

bool create_custom_op(luci::CircleStridedSlice *strided_slice_node)
{
  auto while_out_node = dynamic_cast<luci::CircleWhileOut *>(strided_slice_node->input());
  if (while_out_node == nullptr)
    return false;

  auto while_node = dynamic_cast<luci::CircleWhile *>(while_out_node->input());
  if (while_node == nullptr)
    return false;
  auto input_node = dynamic_cast<luci::CircleNode *>(while_node->input(4));
  auto input_state_node = dynamic_cast<luci::CircleNode *>(while_node->input(3));
  luci::CircleConst *weight_ih = nullptr;
  luci::CircleConst *bias_ih = nullptr;

  luci::CircleConst *weight_hh = nullptr;
  luci::CircleConst *bias_hh = nullptr;

  auto input_size = input_node->dim(input_node->rank() - 1).value();
  auto hidden_size = input_state_node->dim(input_state_node->rank() - 1).value();

  auto body_graph = while_node->body_graph();
  for (auto node : loco::active_nodes(loco::output_nodes(body_graph)))
  {
    auto fc = dynamic_cast<luci::CircleFullyConnected *>(node);
    if (not fc)
      continue;
    auto fc_weights = dynamic_cast<luci::CircleNode *>(fc->weights());

    if (fc_weights->dim(fc->rank() - 1) == input_size)
    {
      weight_ih = dynamic_cast<luci::CircleConst *>(fc->weights());
      bias_ih = dynamic_cast<luci::CircleConst *>(fc->bias());
    }
    if (fc_weights->dim(fc->rank() - 1) == hidden_size)
    {
      weight_hh = dynamic_cast<luci::CircleConst *>(fc->weights());
      bias_hh = dynamic_cast<luci::CircleConst *>(fc->bias());
    }
  }

  assert(weight_hh != nullptr);
  assert(weight_ih != nullptr);
  assert(bias_ih != nullptr);
  assert(bias_hh != nullptr);

  auto weight_ih_cloned = clone_circleconst(weight_ih, strided_slice_node->graph());
  luci::copy_common_attributes(weight_ih, weight_ih_cloned);

  auto weight_hh_cloned = clone_circleconst(weight_hh, strided_slice_node->graph());
  luci::copy_common_attributes(weight_hh, weight_hh_cloned);

  auto bias_ih_cloned = clone_circleconst(bias_ih, strided_slice_node->graph());
  luci::copy_common_attributes(bias_ih, bias_ih_cloned);

  auto bias_hh_cloned = clone_circleconst(bias_hh, strided_slice_node->graph());
  luci::copy_common_attributes(bias_hh, bias_hh_cloned);

  // Create and configure new CircleCustom operation.
  auto fused_gru = while_node->graph()->nodes()->create<luci::CircleCustom>(6, 1);
  auto custom_out = while_node->graph()->nodes()->create<luci::CircleCustomOut>();

  fused_gru->custom_code("custom_gru");
  fused_gru->inputs(0, input_node);
  fused_gru->inputs(1, weight_ih_cloned);
  fused_gru->inputs(2, weight_hh_cloned);
  fused_gru->inputs(3, bias_ih_cloned);
  fused_gru->inputs(4, bias_hh_cloned);
  fused_gru->inputs(5, input_state_node);

  fused_gru->name("gru");

  fused_gru->shape_status(luci::ShapeStatus::VALID);
  fused_gru->rank(2);
  fused_gru->dim(0).set(strided_slice_node->dim(0).value());
  fused_gru->dim(1).set(strided_slice_node->dim(1).value());

  fused_gru->dtype(loco::DataType::FLOAT32);

  custom_out->input(fused_gru);
  custom_out->rank(2);
  custom_out->name("out");
  custom_out->dim(0).set(strided_slice_node->dim(0).value());
  custom_out->dim(1).set(strided_slice_node->dim(1).value());
  custom_out->dtype(loco::DataType::FLOAT32);
  custom_out->shape_status(luci::ShapeStatus::VALID);
  custom_out->index(0);

  replace(strided_slice_node).with(custom_out);

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
    auto strided_slice = dynamic_cast<luci::CircleStridedSlice *>(node);
    if (not strided_slice)
      continue;

    if (create_custom_op(strided_slice))
      changed = true;
  }

  return changed;
}

} // namespace luci
