/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveRedundantTransposePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleOpcode.h>

namespace luci
{

/// @breif Return true if first_prem[second_prem[i]] == i
bool check_perm(const luci::CircleConst *first_perm, const luci::CircleConst *second_perm)
{
  assert(first_perm->rank() == 1);
  assert(second_perm->rank() == 1);
  assert(second_perm->size<loco::DataType::S32>() == first_perm->size<loco::DataType::S32>());
  for (int32_t i = 0; i < static_cast<int32_t>(first_perm->size<loco::DataType::S32>()); i++)
  {
    if (first_perm->at<loco::DataType::S32>(second_perm->at<loco::DataType::S32>(i)) != i)
      return false;
  }
  return true;
}

bool remove_consecutive_transpose_function(luci::CircleNode *node)
{
  if (node->opcode() != luci::CircleOpcode::TRANSPOSE)
    return false;

  auto pred_node = static_cast<luci::CircleNode *>(node->arg(0));
  if (pred_node->opcode() != luci::CircleOpcode::TRANSPOSE)
    return false;
  if (loco::succs(pred_node).size() != 1)
    return false;

  auto pred_perm = dynamic_cast<luci::CircleConst *>(node->arg(1));
  if (pred_perm == nullptr)
    return false;

  auto main_perm = dynamic_cast<luci::CircleConst *>(pred_node->arg(1));
  if (main_perm == nullptr)
    return false;

  auto main_node = static_cast<luci::CircleNode *>(pred_node->arg(0));
  if (check_perm(pred_perm, main_perm))
  {
    replace(node).with(main_node);
    pred_node->drop();
  }
  else
  {
    auto g = main_perm->graph();
    auto new_const_node = g->nodes()->create<luci::CircleConst>();

    new_const_node->dtype(loco::DataType::S32);
    new_const_node->rank(main_perm->rank());
    uint32_t dim_size = 1;
    for (uint32_t i = 0; i < new_const_node->rank(); ++i)
    {
      new_const_node->dim(i) = main_perm->dim(i);
      dim_size *= main_perm->dim(i).value();
    }
    new_const_node->size<loco::DataType::S32>(dim_size);
    new_const_node->shape_status(luci::ShapeStatus::VALID);
    for (uint32_t i = 0; i < main_perm->size<loco::DataType::S32>(); i++)
    {
      new_const_node->at<loco::DataType::S32>(i) =
          pred_perm->at<loco::DataType::S32>(main_perm->at<loco::DataType::S32>(i));
    }
    replace(main_perm).with(new_const_node);
    replace(node).with(pred_node);
  }
  node->drop();
  return true;
}

/**
 *  BEFORE
 *         |
 *   [CircleInput]    [CircleConst]
 *           \              /
 *           [CircleTranspose]  [CircleConst]
 *                   \               /
 *                   [CircleTranspose]
 *                           |
 *
 *  AFTER
 *      <Optional Case>
 *
 *          |                 |               |
 *    [CircleInput]     [CircleConst]         |
 *           \               /           or   |  [Remove all]
 *           [CircleTranspose]                |
 *                   |                        |
 *
 */
bool RemoveRedundantTransposePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = static_cast<luci::CircleNode *>(node);
    if (remove_consecutive_transpose_function(circle_node))
    {
      changed = true;
      break;
    }
  }
  return changed;
}

} // namespace luci
