/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "LayerInfoMap.h"

#include <luci/IR/CircleNode.h>

#include <cassert>

namespace luci
{

LayerInfoMap layer_info_map(loco::Graph *g, std::vector<LayerInfo> &layers_info)
{
  LayerInfoMap info_by_name;

  for (auto &&info : layers_info)
  {
    auto name = info.name;
    bool found = false;
    for (auto node : loco::active_nodes(loco::output_nodes(g)))
    {
      auto cnode = loco::must_cast<luci::CircleNode *>(node);
      if (cnode->name() == name)
      {
        if (info_by_name.find(name) != info_by_name.end())
        {
          throw std::runtime_error("Duplicate layer name " + name +
                                   ". Check layer names in the quantization configuration file.");
        }

        info_by_name[name] = &info;
        found = true;
        continue;
      }
    }

    if (not found)
      throw std::runtime_error("No such layer named " + name +
                               ". Check layer names in the quantization configuration file.");
  }

  assert(info_by_name.size() == layers_info.size()); // FIX_ME_UNLESS

  return info_by_name;
}

} // namespace luci
