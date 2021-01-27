/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/CircleNode.h"

#include <loco.h>

#include <vector>
#include <set>

namespace luci
{

/**
 * @brief Set annotation for circle node id
 */
class CircleNodeID final : public loco::NodeAnnotation
{
public:
  CircleNodeID() = delete;

  CircleNodeID(uint32_t node_id) : _node_id{node_id}
  {
    // Do nothing
  }

public:
  uint32_t node_id(void) const { return _node_id; }
  void node_id(uint32_t id) { _node_id = id; } // Do we need this?

private:
  uint32_t _node_id;
};

/**
 * @brief Set annotation for circle node origin
 */
class CircleNodeOrigin final : public loco::NodeAnnotation
{
public:
  CircleNodeOrigin() = delete;

  CircleNodeOrigin(uint32_t origin) { _origins.insert(origin); }

  CircleNodeOrigin(std::vector<const luci::CircleNode *> origin_nodes)
  {
    for (auto node : origin_nodes)
    {
      assert(node->annot<CircleNodeOrigin>() != nullptr);
      for (auto origin : node->annot<CircleNodeOrigin>()->origins())
        _origins.insert(origin);
    }
  }

public:
  const std::set<uint32_t> &origins(void) const { return _origins; }
  void origins(const std::set<uint32_t> &origins) { _origins = origins; } // Do we need this?

private:
  std::set<uint32_t> _origins;
};

} // namespace luci
