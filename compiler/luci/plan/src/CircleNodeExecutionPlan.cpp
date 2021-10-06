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

#include "luci/Plan/CircleNodeExecutionPlan.h"

#include <loco.h>

#include <stdexcept>
#include <utility>

namespace
{

/**
 * @brief Set annotation for circle node execution plan
 * @note  Once CircleExecutionPlanAnnotation is annotated, it should not be changed.
 *        If CircleExecutionPlanAnnotation is needed to be changed, create
 *        new CircleExecutionPlanAnnotation.
 */
class CircleExecutionPlanAnnotation final : public loco::NodeAnnotation
{
public:
  CircleExecutionPlanAnnotation() = delete;

  explicit CircleExecutionPlanAnnotation(luci::CircleNodeExecutionPlan execution_plan)
    : _execution_plan{std::move(execution_plan)}
  {
    // Do nothing
  }

public:
  const luci::CircleNodeExecutionPlan &execution_plan(void) const { return _execution_plan; }
  // No setter

private:
  luci::CircleNodeExecutionPlan _execution_plan;
};

} // namespace

namespace luci
{

bool has_execution_plan(const luci::CircleNode *circle_node)
{
  return circle_node->annot<CircleExecutionPlanAnnotation>() != nullptr;
}

void add_execution_plan(luci::CircleNode *circle_node,
                        const luci::CircleNodeExecutionPlan &execution_plan)
{
  circle_node->annot<CircleExecutionPlanAnnotation>(nullptr);
  circle_node->annot(std::make_unique<CircleExecutionPlanAnnotation>(execution_plan));
}

luci::CircleNodeExecutionPlan get_execution_plan(const luci::CircleNode *circle_node)
{
  if (!has_execution_plan(circle_node))
    throw std::runtime_error("Cannot find CircleNodeExecutionPlanAnnotation");

  return circle_node->annot<CircleExecutionPlanAnnotation>()->execution_plan();
}

} // namespace luci
