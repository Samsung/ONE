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

#include "luci/Import/Nodes/CircleTile.h"

#include <luci/IR/Nodes/CircleTile.h>

#include <loco.h>

namespace luci
{

bool CircleTileGraphBuilder::validate(const ValidateArgs &args) const
{
  if (!GraphBuilder::validate(args, 2))
    return false;

  auto inputs = args.op.inputs;
  auto outputs = args.op.outputs;
  // Multiples (inputs.at(1)) must be one of the following types
  // int32, int64
  const auto tensors = args.reader.native_tensors();
  const auto &tensor = tensors.at(inputs.at(1));
  switch (tensor->type())
  {
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
      break;
    default:
      return false;
  }

  // Type of input and output must be the same
  if (tensors.at(inputs.at(0))->type() != tensors.at(outputs[0])->type())
    return false;

  return true;
}

CircleNode *CircleTileGraphBuilder::build_node(const circle::OperatorT &,
                                               const std::vector<CircleNode *> &inputs,
                                               loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleTile>();
  node->input(inputs.at(0));
  node->multiples(inputs.at(1));

  return node;
}

} // namespace luci
