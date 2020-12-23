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

#include "luci/Import/Nodes/CircleCast.h"

#include <luci/IR/Nodes/CircleCast.h>

#include <loco.h>

namespace luci
{

bool CircleCastGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;
  if (inputs.size() != 1)
    return false;
  if (outputs.size() != 1)
    return false;

  const auto *options = args.op.builtin_options.AsCastOptions();

  const auto &tensors = args.reader.tensors();

  const auto &tensor_in = tensors.at(inputs[0]);
  if (tensor_in->type != options->in_data_type)
    return false;
  const auto &tensor_out = tensors.at(outputs[0]);
  if (tensor_out->type != options->out_data_type)
    return false;

  return true;
}

CircleNode *CircleCastGraphBuilder::build_node(const circle::OperatorT &op,
                                               const std::vector<CircleNode *> &inputs,
                                               loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleCast>();
  node->x(inputs[0]);

  const auto *options = op.builtin_options.AsCastOptions();
  node->in_data_type(luci_datatype(options->in_data_type));
  node->out_data_type(luci_datatype(options->out_data_type));

  return node;
}

} // namespace luci
