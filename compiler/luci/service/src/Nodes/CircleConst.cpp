/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CircleCloneNode.h"

#include "luci/Service/CircleNodeClone.h"

#include <luci/IR/Nodes/CircleConst.h>

#include <loco.h>
#include <loco/IR/Graph.h>

#include <oops/UserExn.h>

#include <cassert>

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

luci::CircleConst *clone_circleconst(const luci::CircleConst *node, loco::Graph *graph)
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

      case loco::DataType::U4:
        copy_values<loco::DataType::U4>(node, cloned);
        break;

      case loco::DataType::U8:
        copy_values<loco::DataType::U8>(node, cloned);
        break;

      case loco::DataType::S4:
        copy_values<loco::DataType::S4>(node, cloned);
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
        throw oops::UserExn("Unsupported tensor dtype");
    }
  }

  return cloned;
}

} // namespace

namespace luci
{

luci::CircleConst *clone(luci::CircleConst *node)
{
  auto *cloned = clone_circleconst(node, node->graph());

  copy_common_attributes(node, cloned);

  return cloned;
}

} // namespace luci

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::ABC>::visit(const luci::CircleConst *node)
{
  return clone_circleconst(node, _graph);
}

} // namespace luci
