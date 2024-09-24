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

#include "luci/Import/Nodes/CircleRoPE.h"

#include <luci/IR/Nodes/CircleRoPE.h>

#include <loco.h>

#include <iostream>

namespace luci
{

bool CircleRoPEGraphBuilder::validate(const ValidateArgs &args) const
{
  // TODO check dtypes
  return GraphBuilder::validate(args, 3);
}

CircleNode *CircleRoPEGraphBuilder::build_node(const circle::OperatorT &op,
                                               const std::vector<CircleNode *> &inputs,
                                               loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleRoPE>();
  node->input(inputs.at(0));
  node->sin_table(inputs.at(1));
  node->cos_table(inputs.at(2));

  const auto *options = op.builtin_options.AsRoPEOptions();
  node->mode(luci_rope_mode(options->mode));

  return node;
}

} // namespace luci
