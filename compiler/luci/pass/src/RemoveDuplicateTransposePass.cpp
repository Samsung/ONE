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

#include "luci/Pass/RemoveDuplicateTransposePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleOpcode.h>

namespace luci
{

bool check_perm(luci::CircleConst *pred_perm, luci::CircleConst *main_perm)
{
  assert(pred_perm->rank() == 1);
  assert(main_perm->rank() == 1);
  assert(main_perm->size<loco::DataType::S32>() == pred_perm->size<loco::DataType::S32>());
  int32_t idx = 0;
  for (uint32_t i = 0; i < pred_perm->size<loco::DataType::S32>(); i++)
  {
    if (pred_perm->at<loco::DataType::S32>(main_perm->at<loco::DataType::S32>(i)) != idx)
      return false;
    idx = idx + 1;
  }
  return true;
}

bool remove_duplicate_transpose_function(luci::CircleNode *node)
{
  if (node->opcode() != luci::CircleOpcode::TRANSPOSE)
    return false;

  auto pred_node = static_cast<luci::CircleNode *>(node->arg(0));
  auto pred_perm = dynamic_cast<luci::CircleConst *>(node->arg(1));
  if (pred_node->opcode() != luci::CircleOpcode::TRANSPOSE)
    return false;

  auto main_node = static_cast<luci::CircleNode *>(pred_node->arg(0));
  auto main_perm = dynamic_cast<luci::CircleConst *>(pred_node->arg(1));

  if (pred_perm == nullptr || main_perm == nullptr)
    return false;
  if (loco::succs(pred_node).size() != 1)
    return false;
  if (check_perm(pred_perm, main_perm))
  {
    replace(node).with(main_node);
    pred_node->drop();
  }
  else
  {
    std::vector<int32_t> tmp;
    for (uint32_t i = 0; i < main_perm->size<loco::DataType::S32>(); i++)
    {
      tmp.push_back(pred_perm->at<loco::DataType::S32>(main_perm->at<loco::DataType::S32>(i)));
    }
    for (uint32_t i = 0; i < main_perm->size<loco::DataType::S32>(); i++)
    {
      main_perm->at<loco::DataType::S32>(i) = tmp.at(i);
    }
    replace(node).with(pred_node);
  }
  node->drop();
  return true;
}

/**
 *  BEFORE
 *
 *                         |
 *                 [CircleTranspose]
 *                         |
 *                 [CircleTranspose]
 *                         |
 *
 *  AFTER
 *      <Optional Case>
 *
 *             |                      |
 *     [CircleTranspose]     OR       |   Remove Both
 *             |                      |
 *
 */
bool RemoveDuplicateTransposePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = static_cast<luci::CircleNode *>(node);
    if (remove_duplicate_transpose_function(circle_node))
    {
      changed = true;
      break;
    }
  }
  return changed;
}

} // namespace luci
