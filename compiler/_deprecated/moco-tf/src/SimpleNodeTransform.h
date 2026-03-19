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

#ifndef __MOCO_TF_SIMPLE_NODE_TRANSFORM_H__
#define __MOCO_TF_SIMPLE_NODE_TRANSFORM_H__

#include "Transform.h"

namespace moco
{
namespace tf
{

/**
 * @brief Per-Node Transform
 */
template <typename ConcreteNode> struct SimpleNodeTransform : public Transform
{
  SimpleNodeTransform() = default;

  virtual ~SimpleNodeTransform() = default;

  // NOTE Users SHOULD implement this method
  virtual bool transform(ConcreteNode *node) const = 0;

  bool run(loco::Graph *graph) final
  {
    using loco::active_nodes;
    using loco::output_nodes;

    bool changed = false;

    for (auto node : active_nodes(output_nodes(graph)))
    {
      if (auto casted = dynamic_cast<ConcreteNode *>(node))
      {
        if (transform(casted))
        {
          changed = true;
        }
      }
    }

    return changed;
  }
};

} // namespace tf
} // namespace moco

#endif // __MOCO_TF_SIMPLE_NODE_TRANSFORM_H__
