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

#include "luci/Service/CircleShapeSignatureInference.h"

#include <luci/Log.h>

namespace
{

std::ostream &operator<<(std::ostream &os, const luci::ShapeSignature &shape_signature)
{
  os << "[";
  for (uint32_t r = 0; r < shape_signature.rank(); ++r)
  {
    if (r)
      os << ",";
    os << shape_signature.dim(r);
  }
  os << "]";
  return os;
}

} // namespace

namespace luci
{

namespace ssinf
{

bool Rule::infer(const luci::CircleNode *circle_node, ShapeSignature &shape_signature) const
{
  LOGGER(l);

  // There is nothing to check before ShapeSignatureInference.

  Algorithm alg;

  shape_signature = circle_node->accept(&alg);

  VERBOSE(l, 1) << "[luci] Shape Signature( " << circle_node->name() << " )";
  VERBOSE(l, 1) << "    before: " << circle_node->shape_signature();
  VERBOSE(l, 1) << "     after: " << shape_signature;

  return true;
}

} // namespace ssinf

} // namespace luci
