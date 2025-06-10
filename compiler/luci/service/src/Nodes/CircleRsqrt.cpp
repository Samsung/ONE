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

#include "luci/Service/CircleShapeInference.h"

#include "CircleCloneNode.h"
#include "CircleShapeInferenceHelper.h"

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::OPQR>::visit(const luci::CircleRsqrt *)
{
  return _graph->nodes()->create<luci::CircleRsqrt>();
}

namespace sinf
{

loco::TensorShape Algorithm::visit(const luci::CircleRsqrt *node)
{
  const auto input_x = loco::must_cast<CircleNode *>(node->x());
  const auto input_shape = circle_shape(input_x);
  return input_shape;
}

} // namespace sinf
} // namespace luci
