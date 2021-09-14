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

#include "luci/Pass/ExpandBroadcastConstPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Log.h>

#include <type_traits>

namespace
{

luci::CircleConst *create_expanded_constant(luci::CircleConst *node, luci::CircleNode *successor)
{
  LOGGER(l);

  if (successor->rank() != node->rank())
    return nullptr;

  std::vector<uint32_t> broadcast_dims;
  for (uint32_t dim = 0; dim < node->rank(); ++dim)
  {
    if (node->dim(dim) == successor->dim(dim))
      continue;

    if (node->dim(dim) == 1)
      broadcast_dims.push_back(dim);
  }

  if (broadcast_dims.size() != 1 || broadcast_dims.back() != node->rank() - 1)
  {
    WARN(l) << "NYI: Only depth broadcast removal is supported";
    return nullptr;
  }

  auto constant = node->graph()->nodes()->create<luci::CircleConst>();
  constant->name(node->name());
  constant->dtype(node->dtype());
  constant->rank(node->rank());
  constant->shape_status(luci::ShapeStatus::VALID);

  uint32_t node_size = node->size<loco::DataType::FLOAT32>();
  uint32_t constant_size = 1;
  for (uint32_t i = 0; i < successor->rank(); ++i)
  {
    constant->dim(i).set(successor->dim(i).value());
    constant_size *= constant->dim(i).value();
  }
  constant->size<loco::DataType::FLOAT32>(constant_size);

  auto const node_data = &node->at<loco::DataType::FLOAT32>(0);
  auto const constant_data = &constant->at<loco::DataType::FLOAT32>(0);

  auto const successor_depth = successor->dim(successor->rank() - 1).value();
  for (uint32_t d = 0; d < successor_depth; ++d)
    std::copy(node_data, node_data + node_size, constant_data + d * node_size);

  return constant;
}

template <typename N> bool expand_node_input(luci::CircleConst *node, luci::CircleNode *successor)
{
  static_assert(std::is_base_of<luci::CircleNode, N>::value,
                "Successor node should have CircleNode base");

  auto const successor_node = loco::must_cast<N *>(successor);
  auto const successor_x = loco::must_cast<luci::CircleNode *>(successor_node->x());
  auto const successor_y = loco::must_cast<luci::CircleNode *>(successor_node->y());

  luci::CircleConst *expanded_const;

  if (node == successor_x)
  {
    expanded_const = create_expanded_constant(node, successor_y);

    if (expanded_const == nullptr)
      return false;

    successor_node->x(expanded_const);
  }
  else if (node == successor_y)
  {
    expanded_const = create_expanded_constant(node, successor_x);

    if (expanded_const == nullptr)
      return false;

    successor_node->y(expanded_const);
  }

  return true;
}

/**
 * Expand constants following broadcasting rules for binary input nodes (Add, Mul, etc.)
 *
 *    BEFORE
 *
 *    [CircleInput] [CircleConst (H x W x 1)]
 *               |     |
 *             [CircleAdd]
 *
 *    AFTER
 *
 *    [CircleInput] [CircleConst (H x W x D)]
 *               |     |
 *             [CircleAdd]
 */
bool expand_broadcast_const(luci::CircleConst *node)
{
  if (node->dtype() != loco::DataType::FLOAT32)
    return false; // Unsupported data type

  bool changed = false;

  for (auto successor : loco::succs(node))
  {
    auto const circle_successor = loco::must_cast<luci::CircleNode *>(successor);
    switch (circle_successor->opcode())
    {
      case luci::CircleOpcode::ADD:
        if (expand_node_input<luci::CircleAdd>(node, circle_successor))
          changed = true;
        break;
      case luci::CircleOpcode::MUL:
        if (expand_node_input<luci::CircleMul>(node, circle_successor))
          changed = true;
        break;
      case luci::CircleOpcode::DIV:
        if (expand_node_input<luci::CircleDiv>(node, circle_successor))
          changed = true;
        break;
      default:
        break; // Unsupported successor node
    }
  }

  return changed;
}

} // namespace

namespace luci
{

/**
 * Broadcast expanding for Const nodes
 **/
bool ExpandBroadcastConstPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto const_node = dynamic_cast<luci::CircleConst *>(node);
    if (const_node == nullptr)
      continue;

    if (expand_broadcast_const(const_node))
      changed = true;
  }
  return changed;
}

} // namespace luci
