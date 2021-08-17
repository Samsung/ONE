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

#include "luci_interpreter/CircleNodeMemoryPlan.h"

#include <loco.h>

#include <stdexcept>

namespace
{

/**
 * @brief Set annotation for circle node memory plan
 * @note  Once CircleMemoryPlanAnnotation is annotated, it should not be changed.
 *        If CircleMemoryPlanAnnotation is needed to be changed, create
 *        new CircleMemoryPlanAnnotation.
 */
class CircleMemoryPlanAnnotation final : public loco::NodeAnnotation
{
public:
  CircleMemoryPlanAnnotation() = delete;

  CircleMemoryPlanAnnotation(luci::CircleNodeMemoryPlan memory_plan) : _memory_plan{memory_plan}
  {
    // Do nothing
  }

public:
  luci::CircleNodeMemoryPlan memory_plan(void) const { return _memory_plan; }
  // No setter

private:
  luci::CircleNodeMemoryPlan _memory_plan;
};

} // namespace

namespace luci
{

bool has_memory_plan(const luci::CircleNode *circle_node)
{
  return circle_node->annot<CircleMemoryPlanAnnotation>() != nullptr;
}

void add_memory_plan(luci::CircleNode *circle_node, luci::CircleNodeMemoryPlan memory_plan)
{
  circle_node->annot<CircleMemoryPlanAnnotation>(nullptr);
  circle_node->annot(std::make_unique<CircleMemoryPlanAnnotation>(memory_plan));
}

luci::CircleNodeMemoryPlan get_memory_plan(const luci::CircleNode *circle_node)
{
  if (!has_memory_plan(circle_node))
    throw std::runtime_error("Cannot find CircleNodeMemoryPlanAnnotation");

  return circle_node->annot<CircleMemoryPlanAnnotation>()->memory_plan();
}

} // namespace luci

