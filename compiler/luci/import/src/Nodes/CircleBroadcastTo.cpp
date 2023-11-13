/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleBroadcastTo.h"

#include <luci/IR/Nodes/CircleBroadcastTo.h>

#include <loco.h>

namespace luci
{

bool CircleBroadcastToGraphBuilder::validate(const ValidateArgs &args) const
{
  // TODO Support type check
  return GraphBuilder::validate(args, 2);
}

CircleNode *CircleBroadcastToGraphBuilder::build_node(const circle::OperatorT &,
                                                      const std::vector<CircleNode *> &inputs,
                                                      loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleBroadcastTo>();
  node->input(inputs.at(0));
  node->shape(inputs.at(1));

  return node;
}

} // namespace luci
