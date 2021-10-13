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

#include "luci/Import/Nodes/CircleReduceAny.h"

#include <luci/IR/Nodes/CircleReduceAny.h>

namespace luci
{

bool CircleReduceAnyGraphBuilder::validate(const ValidateArgs &args) const
{
  if (!GraphBuilder::validate(args, 2))
    return false;

  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;
  const auto tensors = args.reader.tensors();
  const auto &tensor_0 = tensors.at(inputs.at(0));
  const auto &tensor_1 = tensors.at(inputs.at(1));
  const auto &tensor_o = tensors.at(outputs[0]);

  if (tensor_0->type() != circle::TensorType_BOOL)
    return false;
  if (tensor_o->type() != circle::TensorType_BOOL)
    return false;

  switch (tensor_1->type())
  {
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
      break;
    default:
      return false;
  }

  return true;
}

CircleNode *CircleReduceAnyGraphBuilder::build_node(const circle::OperatorT &op,
                                                    const std::vector<CircleNode *> &inputs,
                                                    loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleReduceAny>();
  node->input(inputs.at(0));
  node->reduction_indices(inputs.at(1));

  const auto *options = op.builtin_options.AsReducerOptions();
  node->keep_dims(options->keep_dims);

  return node;
}

} // namespace luci
