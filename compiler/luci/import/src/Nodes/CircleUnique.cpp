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

#include "luci/Import/Nodes/CircleUnique.h"

#include <luci/IR/Nodes/CircleUnique.h>
#include <luci/IR/Nodes/CircleUniqueOut.h>

#include <loco.h>

namespace luci
{

bool CircleUniqueGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 1)
    return false;

  if (args.op.outputs.size() != 2)
    return false;

  return true;
}

CircleNode *CircleUniqueGraphBuilder::build_node(const BuildNodeArgs &bna) const
{
  auto node = bna.context->graph()->nodes()->create<CircleUnique>();

  node->input(bna.input_nodes[0]);

  const auto *options = bna.op.builtin_options.AsUniqueOptions();
  node->output_type(luci_datatype(options->idx_out_type));

  return node;
}

CircleNode *CircleUniqueGraphBuilder::build_out(const BuildOutArgs &boa) const
{
  auto *nodeout = boa.node->graph()->nodes()->create<CircleUniqueOut>();

  nodeout->input(boa.node);
  nodeout->index(boa.index);

  return nodeout;
}

} // namespace luci
