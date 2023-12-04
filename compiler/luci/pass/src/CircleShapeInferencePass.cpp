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

#include "helpers/InferenceCandidates.h"
#include "helpers/Shape.h"

#include "luci/Pass/CircleShapeInferencePass.h"

#include <luci/Service/CircleShapeInference.h>

#include <loco.h>

namespace luci
{

bool CircleShapeInferencePass::run(luci::Module *m)
{
  bool changed = false;

  for (size_t g = 0; g < m->size(); ++g)
  {
    if (run(m->graph(g)))
      changed = true;
  }

  return changed;
}

bool CircleShapeInferencePass::run(loco::Graph *g)
{
  luci::sinf::Rule shape_infer_rule;
  bool changed = false;

  for (auto node : inference_candidates(g))
  {
    loco::TensorShape shape;
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    if (shape_infer_rule.infer(circle_node, shape) && !is_same_shape(circle_node, shape))
    {
      circle_node->rank(shape.rank());
      for (uint32_t i = 0; i < shape.rank(); ++i)
        circle_node->dim(i) = shape.dim(i);

      circle_node->shape_status(luci::ShapeStatus::VALID);

      changed = true;
    }
  }

  return changed;
}

} // namespace luci
