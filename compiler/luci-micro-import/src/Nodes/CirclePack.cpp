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

#include "luci/Import/Nodes/CirclePack.h"

#include <luci/IR/Nodes/CirclePack.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CirclePackGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;
  const auto *options = args.op.builtin_options.AsPackOptions();

  if (options->values_count < 1)
    return false;

  if (inputs.size() != static_cast<uint32_t>(options->values_count))
    return false;

  if (outputs.size() != 1)
    return false;

  return true;
}

CircleNode *CirclePackGraphBuilder::build_node(const circle::OperatorT &op,
                                               const std::vector<CircleNode *> &inputs,
                                               loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CirclePack>(inputs.size());
  for (uint32_t i = 0; i < inputs.size(); ++i)
  {
    node->values(i, inputs[i]);
  }

  const auto *options = op.builtin_options.AsPackOptions();
  node->axis(options->axis);

  return node;
}

} // namespace luci
