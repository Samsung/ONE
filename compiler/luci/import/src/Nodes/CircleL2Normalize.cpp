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

#include "luci/Import/Nodes/CircleL2Normalize.h"

#include <luci/IR/Nodes/CircleL2Normalize.h>

#include <loco.h>

namespace luci
{

bool CircleL2NormalizeGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  if (inputs.size() != 1)
  {
    return false;
  }

  if (outputs.size() != 1)
  {
    return false;
  }

  return true;
}

CircleNode *CircleL2NormalizeGraphBuilder::build_node(const circle::OperatorT &op,
                                                      const std::vector<CircleNode *> &inputs,
                                                      loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleL2Normalize>();
  node->x(inputs.at(0));
  const auto *options = op.builtin_options.AsL2NormOptions();
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));

  return node;
}

} // namespace luci
