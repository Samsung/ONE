/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "locoex/Service/COpShapeInferenceRule.h"

#include "locoex/COpDialect.h"
#include "locoex/COpNode.h"
#include "locoex/COpCall.h"

#include <loco/Service/ShapeInference.h>

#include <cassert>

namespace locoex
{

bool COpShapeInferenceRule::recognize(const loco::Dialect *d) const
{
  return COpDialect::get() == d;
}

bool COpShapeInferenceRule::infer(const loco::Node *node, loco::NodeShape &shape) const
{
  assert(node->dialect() == COpDialect::get());
  assert(dynamic_cast<const COpNode *>(node) != nullptr);

  auto cop_call = dynamic_cast<const COpCall *>(node);

  // Note that the shape of custom op is considered as TensorShape
  // TODO Decide how to deal with this shape error cases
  for (uint32_t n = 0; n < cop_call->arity(); n++)
    if (loco::shape_get(cop_call->input(n)).domain() != loco::Domain::Tensor)
      throw std::runtime_error("Input of custom op must belong to Tensor domain.");

  loco::TensorShape out_shape;

  out_shape.rank(cop_call->rank());
  for (uint32_t d = 0; d < cop_call->rank(); d++)
    out_shape.dim(d) = cop_call->dim(d);

  shape.set(out_shape);

  return true;
}

} // namespace locoex
