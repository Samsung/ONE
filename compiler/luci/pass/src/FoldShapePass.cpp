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

#include "luci/Pass/FoldShapePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

namespace
{

template <loco::DataType OutType> luci::CircleConst *folding(luci::CircleShape *shape)
{
  auto input_node = loco::must_cast<luci::CircleNode *>(shape->input());
  auto name = input_node->name();
  assert(name.length() > 0);
  auto shape_status = input_node->shape_status();
  if (shape_status != luci::ShapeStatus::VALID)
    return nullptr;
  auto rank = input_node->rank();
  // TODO support rank == 0 when possible
  if (rank == 0)
    return nullptr;
  for (uint32_t i = 0; i < rank; i++)
  {
    auto dim = input_node->dim(i);
    if (!dim.known())
      return nullptr;
  }

  auto folded_shape = input_node->graph()->nodes()->create<luci::CircleConst>();
  folded_shape->name(name + "_ConstShape");
  folded_shape->dtype(OutType);
  folded_shape->rank(1);
  folded_shape->dim(0).set(rank);
  luci::add_origin(folded_shape, luci::get_origin(shape));

  folded_shape->size<OutType>(rank);
  for (uint32_t i = 0; i < rank; i++)
    folded_shape->at<OutType>(i) = input_node->dim(i).value();

  return folded_shape;
}

// Fold Shape to const if the input shape is static
template <loco::DataType OutType> bool fold_shape(luci::CircleShape *shape)
{
  auto folded_shape = folding<OutType>(shape);
  if (not folded_shape)
    return false;

  loco::replace(shape).with(folded_shape);

  return true;
}

} // namespace

namespace luci
{

/**
 * BEFORE
 *
 *     [CircleNode]
 *           |
 *     [CircleShape]
 *           |
 *     [CircleNode]
 *
 * AFTER
 *
 *     [CircleConst]  [CircleNode]
 *           |
 *     [CircleNode]
 *
 */
bool FoldShapePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto shape = dynamic_cast<luci::CircleShape *>(node))
    {
      auto out_type = shape->out_type();
      switch (out_type)
      {
        case loco::DataType::S32:
          if (fold_shape<loco::DataType::S32>(shape))
            changed = true;
          break;
        case loco::DataType::S64:
          if (fold_shape<loco::DataType::S64>(shape))
            changed = true;
          break;
        default:
          break;
      }
    }
  }

  return changed;
}

} // namespace luci
