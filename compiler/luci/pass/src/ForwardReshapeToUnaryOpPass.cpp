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

#include "luci/Pass/ForwardReshapeToUnaryOpPass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>
#include <luci/Service/CircleShapeInference.h>
#include <luci/Service/Nodes/CircleConst.h>

namespace
{

luci::CircleReshape *as_reshape(loco::Node *node)
{
  return dynamic_cast<luci::CircleReshape *>(node);
}

bool forward_reshape(luci::CircleReshape *reshape, luci::CircleNeg *neg)
{
  assert(reshape != nullptr);
  assert(neg != nullptr);

  luci::CircleConst *cloned_shape = nullptr;
  const auto reshape_shape = dynamic_cast<luci::CircleConst *>(reshape->shape());
  // only support CircleConst for now
  if (reshape_shape == nullptr)
    return false;

  auto dtype = reshape_shape->dtype();
  if (dtype != loco::DataType::S32 && dtype != loco::DataType::S64)
    return false;
  cloned_shape = luci::clone(reshape_shape);

  loco::Graph *graph = neg->graph();
  // create reshape placed after neg
  luci::CircleReshape *new_reshape = graph->nodes()->create<luci::CircleReshape>();
  auto ns_rank = reshape->newShape()->rank();
  new_reshape->newShape()->rank(ns_rank);
  for (uint32_t r = 0; r < ns_rank; ++r)
    new_reshape->newShape()->dim(r) = reshape->newShape()->dim(r);

  new_reshape->shape(cloned_shape);

  // reconnect network
  loco::replace(neg).with(new_reshape);
  neg->x(reshape->tensor());
  new_reshape->tensor(neg);

  // need to reset shape as it leaped over Reshape
  // TODO Remove loco::shape_erase()
  loco::shape_erase(neg);
  neg->shape_status(luci::ShapeStatus::UNDEFINED);

  return true;
}

class ForwardReshape final : public luci::CircleNodeMutableVisitor<bool>
{
protected:
  bool visit(luci::CircleNode *node)
  {
    LOGGER(l);
    INFO(l) << "ForwardReshape: Unsupported operator: " << node->name() << std::endl;
    return false;
  }

  bool visit(luci::CircleNeg *node)
  {
    auto reshape = as_reshape(node->x());
    if (reshape == nullptr)
      return false;
    return forward_reshape(reshape, node);
  }

  // TODO add more unary operators
};

} // namespace

namespace luci
{

/**
 * BEFORE
 *                       |
 *                  [CircleNode]  [CircleConst]
 *                       |       /
 *                 [CircleReshape]
 *                /      |
 *     [CircleNode]  [(UnaryOp)]
 *          |            |     \
 *          |            |      [CircleNode]
 *          |            |           |
 *
 *   UnaryOp: CircleNeg, ...
 *
 * AFTER
 *                       |
 *   [CircleConst]  [CircleNode]
 *         |       /     |
 *  [CircleReshape] [(UnaryOp)] [CircleConst]
 *         |             |      /
 *   [CircleNode] [CircleReshape]
 *         |             |      \
 *         |             |       [CircleNode]
 *         |             |            |
 *
 *   Note: new [CircleReshape] after [(UnaryOp)] added
 */
bool ForwardReshapeToUnaryOpPass::run(loco::Graph *g)
{
  bool changed = false;
  ForwardReshape forward;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (circle_node->accept(&forward))
      changed = true;
  }
  return changed;
}

} // namespace luci
