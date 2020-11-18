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

namespace luci
{

bool ShapeSignatureInferencePass::run(luci::Module *m)
{
  bool changed = false;

  for (size_t g = 0;g < m->size();++g)
  {
    changed = changed || run(m->graph(g));
  }

  return changed;
}

bool ShapeSignatureInferencePass::run(loco::Graph *g)
{
  luci::ssinf::Rule signature_inference_rule;
  bool changed = false;

  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    luci::ShapeSignature shape_signature;

    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (signature_inference_rule.infer(circle_node, shape_signature))
    {
      if (!(circle_node->shape_signature() == shape_signature))
      {
        circle_node->shape_signature(shape_signature);
        changed = true;
      }
    }
  }

  return changed;
}

} // namespace luci
