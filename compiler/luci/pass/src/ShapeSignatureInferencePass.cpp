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

#include "luci/Pass/ShapeSignatureInferencePass.h"

#include <luci/IR/CircleShapeSignature.h>
#include <luci/Service/CircleShapeSignatureInference.h>

#include <loco.h>

namespace
{

bool is_same_signature(luci::ShapeSignature s1, luci::ShapeSignature s2)
{
  if (s1.rank() != s2.rank())
    return false;

  for (uint32_t i = 0; i < s1.rank(); ++i)
    if (s1.dim(i) != s2.dim(i))
      return false;

  return true;
}

} // namespace

namespace luci
{

bool ShapeSignatureInferencePass::run(loco::Graph *g)
{
  luci::ssinf::Rule signature_rule;
  bool changed = false;

  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    luci::ShapeSignature shape_signature;

    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (signature_rule.infer(circle_node, shape_signature))
    {
      if (!is_same_signature(circle_node->shape_signature(), shape_signature))
      {
        circle_node->shape_signature(shape_signature);
        changed = true;
      }
    }
  }

  return changed;
}

} // namespace luci
