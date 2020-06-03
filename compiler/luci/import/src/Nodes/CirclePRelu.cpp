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

#include "luci/Import/Nodes/CirclePRelu.h"

#include <luci/IR/Nodes/CirclePRelu.h>

#include <loco.h>

namespace luci
{

bool CirclePReluGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 2)
    return false;

  if (args.op.outputs.size() != 1)
    return false;

  return true;
}

CircleNode *CirclePReluGraphBuilder::build_node(const circle::OperatorT &,
                                                const std::vector<CircleNode *> &inputs,
                                                loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CirclePRelu>();
  node->input(inputs[0]);
  node->alpha(inputs[1]);

  // PRelu options are empty

  return node;
}

} // namespace luci
