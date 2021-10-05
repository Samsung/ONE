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

#ifndef __LUCI_CIRCLE_NODE_EXECUTION_PLAN_H__
#define __LUCI_CIRCLE_NODE_EXECUTION_PLAN_H__

#include <luci/IR/CircleNode.h>

#include <utility>

namespace luci
{

class CircleNodeExecutionPlan
{
public:
  CircleNodeExecutionPlan() = delete;

  CircleNodeExecutionPlan(uint32_t order_in_plan, std::vector<uint32_t> offsets)
  {
    _order_in_plan = order_in_plan;
    _offsets = std::move(offsets);
  }

  uint32_t order_in_plan(void) const { return _order_in_plan; }
  void order_in_plan(const uint32_t &order_in_plan) { _order_in_plan = order_in_plan; }

  std::vector<uint32_t> offsets(void) const { return _offsets; }
  void offsets(const std::vector<uint32_t> &offsets) { _offsets = offsets; }

private:
  uint32_t _order_in_plan;
  std::vector<uint32_t> _offsets;
};

bool has_execution_plan(const luci::CircleNode *circle_node);

void add_execution_plan(luci::CircleNode *circle_node,
                        const luci::CircleNodeExecutionPlan &execution_plan);

luci::CircleNodeExecutionPlan get_execution_plan(const luci::CircleNode *circle_node);

} // namespace luci

#endif // __LUCI_CIRCLE_NODE_EXECUTION_PLAN_H__
