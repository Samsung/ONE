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

luci::CircleNode *CloneNodeLet<CN::OPQR>::visit(const luci::CirclePad *)
{
  return _graph->nodes()->create<luci::CirclePad>();
}

namespace sinf
{

loco::TensorShape Algorithm::visit(const luci::CirclePad *node)
{
  auto paddings = loco::must_cast<const luci::CircleNode *>(node->paddings());
  auto circle_input = loco::must_cast<const luci::CircleNode *>(node->input());
  auto input_shape = circle_shape(circle_input);
  return pad_shape(input_shape, paddings);
}

} // namespace sinf
} // namespace luci
