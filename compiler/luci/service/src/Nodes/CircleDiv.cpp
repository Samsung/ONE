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

#include <luci/Service/CircleShapeInference.h>

#include "CircleCloneNode.h"

#include "CircleShapeInferenceHelper.h"

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::DEF>::visit(const luci::CircleDiv *node)
{
  if (node->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return nullptr;

  auto *cloned = _graph->nodes()->create<luci::CircleDiv>();
  cloned->fusedActivationFunction(node->fusedActivationFunction());
  return cloned;
}

namespace sinf
{

loco::TensorShape Algorithm::visit(const luci::CircleDiv *node)
{
  const auto x = loco::must_cast<luci::CircleNode *>(node->x());
  const auto y = loco::must_cast<luci::CircleNode *>(node->y());

  const auto x_shape = circle_shape(x);
  const auto y_shape = circle_shape(y);

  auto output_shape = broadcast_shape(x_shape, y_shape);

  return output_shape;
}

} // namespace sinf
} // namespace luci
