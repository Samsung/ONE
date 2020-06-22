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

#include "luci/Import/Nodes/CircleAddN.h"

#include <luci/IR/Nodes/CircleAdd.h>

#include <loco.h>

namespace luci
{

bool CircleAddNGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() < 1)
    return false;

  if (args.op.outputs.size() != 1)
    return false;

  return true;
}

CircleNode *CircleAddNGraphBuilder::build_node(const circle::OperatorT &,
                                               const std::vector<CircleNode *> &inputs,
                                               loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleAddN>(inputs.size());
  for (uint32_t i = 0; i < inputs.size(); ++i)
  {
    node->inputs(i, inputs[i]);
  }

  return node;
}

} // namespace luci
