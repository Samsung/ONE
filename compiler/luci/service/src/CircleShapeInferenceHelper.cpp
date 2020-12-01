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

#include "luci/Service/CircleShapeInferenceHelper.h"

namespace luci
{

namespace sinf
{

loco::TensorShape circle_shape(const luci::CircleNode *node)
{
  loco::TensorShape shape;
  shape.rank(node->rank());
  for (uint32_t r = 0; r < node->rank(); ++r)
    shape.dim(r) = loco::Dimension(node->dim(r).value());
  return shape;
}

loco::TensorShape input_arg_shape(const luci::CircleNode *node, unsigned int index)
{
  if (node->arity() <= index)
    throw std::runtime_error("Arity index out of range");

  auto input_node = loco::must_cast<luci::CircleNode *>(node->arg(index));
  return circle_shape(input_node);
}

loco::TensorShape signature_to_shape(const luci::ShapeSignature &signature)
{
  loco::TensorShape shape;
  shape.rank(signature.rank());
  for (uint32_t d = 0; d < signature.rank(); ++d)
    shape.dim(d) = (signature.dim(d) == -1) ? 1 : signature.dim(d);
  return shape;
}

} // namespace sinf

} // namespace luci
