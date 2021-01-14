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

#include "CircleShapeInferenceHelper.h"

namespace luci
{

loco::NodeShape shape_get(const loco::Node *node)
{
  assert(shape_known(node));
  return loco::NodeShape{sinf::circle_shape(loco::must_cast<const luci::CircleNode *>(node))};
}

bool shape_known(const loco::Node *node)
{
  return loco::must_cast<const luci::CircleNode *>(node)->shape_status() !=
         luci::ShapeStatus::UNDEFINED;
}

} // namespace luci

namespace luci
{
namespace sinf
{

loco::TensorShape circle_shape(const luci::CircleNode *node)
{
  loco::TensorShape shape;
  shape.rank(node->rank());
  for (uint32_t r = 0; r < node->rank(); ++r)
    shape.dim(r) = node->dim(r);
  return shape;
}

} // namespace sinf
} // namespace luci
