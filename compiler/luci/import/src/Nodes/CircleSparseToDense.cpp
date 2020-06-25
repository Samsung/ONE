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

#include "luci/Import/Nodes/CircleSparseToDense.h"

#include <luci/IR/Nodes/CircleSparseToDense.h>

#include <loco.h>

namespace luci
{

bool CircleSparseToDenseGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 4)
    return false;

  return true;
}

CircleNode *CircleSparseToDenseGraphBuilder::build_node(const circle::OperatorT &op,
                                                        const std::vector<CircleNode *> &inputs,
                                                        loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleSparseToDense>();
  node->indices(inputs[0]);
  node->output_shape(inputs[1]);
  node->values(inputs[2]);
  node->default_value(inputs[3]);

  const auto *options = op.builtin_options.AsSparseToDenseOptions();
  node->validate_indices(options->validate_indices);

  return node;
}

} // namespace luci
