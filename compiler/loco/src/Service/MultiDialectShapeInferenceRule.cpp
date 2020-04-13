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

#include "loco/Service/MultiDialectShapeInferenceRule.h"
#include "loco/Service/ShapeInferenceRule.h"

#include <loco/IR/Dialect.h>
#include <loco/IR/Node.h>
#include <loco/IR/NodeShape.h>

#include <cassert>

namespace loco
{

bool MultiDialectShapeInferenceRule::recognize(const Dialect *d) const
{
  const auto found = _rules.find(d);

  if (found == _rules.cend())
    return false;

  auto rule = found->second;
  auto result = rule->recognize(d);

  return result;
}

bool MultiDialectShapeInferenceRule::infer(const Node *node, NodeShape &shape) const
{
  const auto found = _rules.find(node->dialect());

  if (found == _rules.cend())
    return false;

  auto rule = found->second;
  if (rule->infer(node, shape))
    return true;

  return false;
}

MultiDialectShapeInferenceRule &MultiDialectShapeInferenceRule::bind(const Dialect *d,
                                                                     const ShapeInferenceRule *rule)
{
  assert(_rules.find(d) == _rules.end());
  assert(rule->recognize(d));

  _rules[d] = rule;

  return (*this);
}

} // namespace loco
