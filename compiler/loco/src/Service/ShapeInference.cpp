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

#include "loco/Service/ShapeInference.h"
#include "loco/IR/Algorithm.h"

#include <cassert>
#include <memory>

namespace
{

bool inputs_shape_ready(loco::Node *node)
{
  assert(node != nullptr);

  for (uint32_t arity = 0; arity < node->arity(); ++arity)
  {
    if (!loco::ShapeInference::known(node->arg(arity)))
    {
      return false;
    }
  }
  return true;
}

} // namespace

//
// Infrastructure
//
namespace
{

struct ShapeAnnotation : public loco::NodeAnnotation
{
public:
  ShapeAnnotation(const loco::NodeShape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  const loco::NodeShape &shape(void) const { return _shape; }

private:
  loco::NodeShape _shape;
};

} // namespace

namespace loco
{

bool ShapeInferenceSession::to(Graph *g) const
{
  assert(_rule->support(ShapeInferenceRule::API::V1) && "API v1 is unavailable");

  bool changed = false;

  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    if (_rule->recognize(node->dialect()))
    {
      loco::NodeShape shape;

      if (!shape_known(node) && inputs_shape_ready(node))
      {
        if (_rule->infer(node, shape))
        {
          node->annot(std::make_unique<ShapeAnnotation>(shape));
          changed = true;
        }
      }
    }
  }

  return changed;
}

bool ShapeInference::known(const Node *node) { return node->annot<ShapeAnnotation>() != nullptr; }

NodeShape ShapeInference::get(const Node *node)
{
  assert(known(node));
  return node->annot<ShapeAnnotation>()->shape();
}

void ShapeInference::erase(Node *node) { node->annot<ShapeAnnotation>(nullptr); }

} // namespace loco
