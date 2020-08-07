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

#include "luci/Import/Nodes/CircleShape.h"

#include <luci/IR/Nodes/CircleShape.h>

#include <loco.h>

namespace luci
{

bool CircleShapeGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;
  if (inputs.size() != 1)
    return false;
  if (outputs.size() != 1)
    return false;

  // TODO check shape, dtype

  return true;
}

CircleNode *CircleShapeGraphBuilder::build_node(const circle::OperatorT &op,
                                                const std::vector<CircleNode *> &inputs,
                                                loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleShape>();
  node->input(inputs.at(0));

  const auto *options = op.builtin_options.AsShapeOptions();
  node->out_type(luci_datatype(options->out_type));

  return node;
}

} // namespace luci
