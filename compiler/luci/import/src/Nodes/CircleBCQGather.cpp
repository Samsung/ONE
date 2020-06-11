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

#include "luci/Import/Nodes/CircleBCQGather.h"

#include <luci/IR/Nodes/CircleBCQGather.h>

#include <loco.h>

namespace luci
{

bool CircleBCQGatherGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 3)
    return false;

  return true;
}

CircleNode *CircleBCQGatherGraphBuilder::build_node(const circle::OperatorT &op,
                                                    const std::vector<CircleNode *> &inputs,
                                                    loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleBCQGather>();

  node->input_scales(inputs[0]);
  node->input_binary(inputs[1]);
  node->indices(inputs[2]);

  const auto *options = op.builtin_options.AsBCQGatherOptions();
  node->input_hidden_size(options->input_hidden_size);
  node->axis(options->axis);

  return node;
}

} // namespace luci
