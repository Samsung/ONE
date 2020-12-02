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

#include "luci/Pass/CopyLocoItemsToCirclePass.h"

#include <loco/Service/ShapeInference.h>
#include <loco/Service/TypeInference.h>

#include <luci/IR/CircleNodes.h>

#include <loco.h>

namespace
{

bool has_same_shape(luci::CircleNode *node, loco::TensorShape shape)
{
  if (node->rank() != shape.rank())
    return false;

  for (uint32_t i = 0; i < shape.rank(); ++i)
    if (!(node->dim(i) == shape.dim(i)))
      return false;

  return true;
}

} // namespace

namespace luci
{

bool CopyLocoItemsToCirclePass::run(luci::Module *m)
{
  bool changed = false;

  for (size_t g = 0; g < m->size(); ++g)
  {
    if (run(m->graph(g)))
      changed = true;
  }

  return changed;
}

bool CopyLocoItemsToCirclePass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::all_nodes(g))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (loco::shape_known(node))
    {
      auto loco_shape = loco::shape_get(node).as<loco::TensorShape>();

      // When shape of loco is copied to circle node, ShapeSignature should be applied.
      loco::TensorShape new_shape;
      new_shape.rank(loco_shape.rank());
      for (uint32_t i = 0; i < loco_shape.rank(); ++i)
      {
        if (circle_node->shape_signature().rank() > 0 &&
            circle_node->shape_signature().dim(i) == -1)
          new_shape.dim(i) = 1;
        else
          new_shape.dim(i) = loco_shape.dim(i);
      }

      if (circle_node->shape_status() == luci::ShapeStatus::UNDEFINED ||
          !has_same_shape(circle_node, new_shape))
      {
        circle_node->rank(new_shape.rank());
        for (uint32_t i = 0; i < new_shape.rank(); ++i)
          circle_node->dim(i) = new_shape.dim(i);

        if (circle_node->shape_status() == luci::ShapeStatus::UNDEFINED)
          circle_node->shape_status(luci::ShapeStatus::VALID);

        changed = true;
      }
    }

    if (loco::dtype_known(node))
    {
      if (loco::dtype_get(node) != circle_node->dtype())
      {
        circle_node->dtype(loco::dtype_get(node));
        changed = true;
      }
    }
  }

  return changed;
}

} // namespace luci
