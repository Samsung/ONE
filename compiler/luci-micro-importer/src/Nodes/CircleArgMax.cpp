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

#include "luci/Import/Nodes/CircleArgMax.h"

#include <luci/IR/Nodes/CircleArgMax.h>

#include <loco.h>

namespace luci
{

bool CircleArgMaxGraphBuilder::validate(const ValidateArgs &args) const
{
  return GraphBuilder::validate(args, 2);
}

CircleNode *CircleArgMaxGraphBuilder::build_node(const circle::OperatorT &op,
                                                 const std::vector<CircleNode *> &inputs,
                                                 loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleArgMax>();
  node->input(inputs.at(0));
  node->dimension(inputs.at(1));

  const auto *options = op.builtin_options.AsArgMaxOptions();
  node->output_type(luci_datatype(options->output_type));

  return node;
}

} // namespace luci
