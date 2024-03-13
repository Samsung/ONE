/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleGRU.h"

#include <luci/IR/Nodes/CircleHardSwish.h>

#include <loco.h>

namespace luci
{

bool CircleGRUGraphBuilder::validate(const ValidateArgs &args) const
{
  return GraphBuilder::validate(args, 6);
}

CircleNode *CircleGRUGraphBuilder::build_node(const circle::OperatorT &,
                                              const std::vector<CircleNode *> &inputs,
                                              loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleGRU>();
  node->input(inputs.at(0));
  node->hidden_hidden(inputs.at(1));
  node->hidden_hidden_bias(inputs.at(2));
  node->hidden_input(inputs.at(3));
  node->hidden_input_bias(inputs.at(4));
  node->state(inputs.at(5));

  return node;
}

} // namespace luci
