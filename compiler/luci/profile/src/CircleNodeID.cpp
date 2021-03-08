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

#include "luci/Profile/CircleNodeID.h"

#include <loco.h>

#include <stdexcept>

namespace
{

/**
 * @brief Set annotation for circle node id
 * @note  Once CircleNodeID is annotated, it should not be changed.
 *        If CircleNodeID is needed to be changed, create new CircleNodeID.
 */
class CircleNodeIDAnnotation final : public loco::NodeAnnotation
{
public:
  CircleNodeIDAnnotation() = delete;

  CircleNodeIDAnnotation(luci::CircleNodeID node_id) : _node_id{node_id}
  {
    // Do nothing
  }

public:
  luci::CircleNodeID node_id(void) const { return _node_id; }
  // No setter

private:
  luci::CircleNodeID _node_id;
};

} // namespace

namespace luci
{

bool has_node_id(const luci::CircleNode *circle_node)
{
  return circle_node->annot<CircleNodeIDAnnotation>() != nullptr;
}

void set_node_id(luci::CircleNode *circle_node, luci::CircleNodeID id)
{
  circle_node->annot<CircleNodeIDAnnotation>(nullptr);
  circle_node->annot(std::make_unique<CircleNodeIDAnnotation>(id));
}

luci::CircleNodeID get_node_id(const luci::CircleNode *circle_node)
{
  if (!has_node_id(circle_node))
    throw std::runtime_error("Cannot find CircleNodeID");

  return circle_node->annot<CircleNodeIDAnnotation>()->node_id();
}

} // namespace luci
