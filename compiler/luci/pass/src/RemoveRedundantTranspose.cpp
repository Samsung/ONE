/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/RemoveRedundantTransposePass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

/// @brief Return true if first_perm[second_perm[i]] == i
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
  auto target_node = dynamic_cast<luci::CircleTranspose *>(node);
  if (target_node == nullptr)
    return false;
  auto pred_node = dynamic_cast<luci::CircleTranspose *>(target_node->a());
  if (pred_node == nullptr)
    return false;

  auto pred_perm = dynamic_cast<luci::CircleConst *>(target_node->perm());
  if (pred_perm == nullptr)
    return false;

  auto main_perm = dynamic_cast<luci::CircleConst *>(pred_node->perm());
  if (main_perm == nullptr)
    return false;

  auto main_node = loco::must_cast<luci::CircleNode *>(pred_node->a());
  if (check_perm(pred_perm, main_perm))
  {
    replace(node).with(main_node);
  }
  else
  {
    auto g = main_node->graph();

    auto new_const_node = g->nodes()->create<luci::CircleConst>();
    new_const_node->dtype(loco::DataType::S32);
    new_const_node->rank(1);
    new_const_node->dim(0) = main_perm->dim(0);
    new_const_node->size<loco::DataType::S32>(main_perm->dim(0).value());
    new_const_node->shape_status(luci::ShapeStatus::VALID);
    for (uint32_t i = 0; i < main_perm->size<loco::DataType::S32>(); i++)
    {
      new_const_node->at<loco::DataType::S32>(i) =
          pred_perm->at<loco::DataType::S32>(main_perm->at<loco::DataType::S32>(i));
    }

    // Create New Transpose Node
    auto new_transpose_node = g->nodes()->create<luci::CircleTranspose>();
    new_transpose_node->dtype(target_node->dtype());
    new_transpose_node->a(main_node);
    new_transpose_node->perm(new_const_node);

    replace(node).with(new_transpose_node);
  }
  return true;
}

} // namespace

namespace luci
{
/**
 *  BEFORE
 *         |
 *   [CircleNode]     [CircleConst]
 *    (main_node)      (main_perm)
 *         \               /
 *         [CircleTranspose]  [CircleConst]
 *            (pred_node)      (pred_perm)
 *                 \               /
 *                 [CircleTranspose]
 *                   (target_node)
 *                         |
 *
 *  AFTER
 *      <Optional Case>
 *
 *          |                 |                   |
 *    [CircleNode]      [CircleConst]             |
 *     (main_node)     (new_const_node)           |
 *           \               /           or  [CircleNode]
 *           [CircleTranspose]                (main_node)
 *              (pred_node)                       |
 *                   |                            |
 *
 */
bool RemoveRedundantTransposePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (remove_consecutive_transpose_function(circle_node))
    {
      changed = true;
      break;
    }
  }
  return changed;
}

} // namespace luci
