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

#include "luci/Pass/RemoveConstDeadNodePass.h"

#include <luci/IR/CircleNodes.h>
#include <loco/IR/Graph.h>

namespace
{

} // namespace

namespace luci
{

bool RemoveConstDeadNodePass::run(loco::Graph *g)
{
  bool changed = false;

  auto graph_outputs = g->outputs();
  auto size = graph_outputs->size();

  for (uint32_t s = 0; s < size; s++)
  {
    auto output = graph_outputs->at(s);

    if (!output->name().compare("node name"))
    {
      auto index = output->index();
      // erase circle output
      auto output_nodes = loco::output_nodes(g);
      for (auto node : output_nodes)
      {
        if (auto circle_output = dynamic_cast<luci::CircleOutput *>(node))
        {
          if (circle_output->index() != index)
            continue;
          circle_output->index(static_cast<int64_t>(-1));
          break;
        }
      }
      // erase graph output
      graph_outputs->erase_output(output);
      size--;

      changed = true;
    }
  }

  return changed;
}

} // namespace luci
