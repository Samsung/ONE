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

#include "CircleShapeInferenceRule.h"

#include "Dialect/IR/CircleNodes.h"
#include "Dialect/IR/CircleDialect.h"
#include "Dialect/IR/CircleNodeVisitor.h"

#include "Check.h"

#include <cassert>

namespace
{

/**
 * @brief Class to infer the shape of CircleNode
 *
 * @note All CircleNode's inputs and outputs are always loco::Domain::Tensor
 */
class ShapeInferenceAlgorithm final : public locoex::CircleNodeVisitor<loco::NodeShape>
{
public:
  loco::NodeShape visit(const locoex::CircleInstanceNorm *node) final
  {
    auto input_shape = loco::shape_get(node->input()).as<loco::TensorShape>();

    return loco::NodeShape{input_shape};
  }
};

} // namespace

namespace locoex
{

bool CircleShapeInferenceRule::recognize(const loco::Dialect *d) const
{
  return CircleDialect::get() == d;
}

bool CircleShapeInferenceRule::infer(const loco::Node *node, loco::NodeShape &shape) const
{
  assert(node->dialect() == CircleDialect::get());
  assert(dynamic_cast<const CircleNode *>(node) != nullptr);

  ShapeInferenceAlgorithm alg;
  shape = dynamic_cast<const CircleNode *>(node)->accept(&alg);

  return true;
}

} // namespace locoex
