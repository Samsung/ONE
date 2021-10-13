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

#include "luci/Import/Nodes/CircleReverseV2.h"

#include <luci/IR/Nodes/CircleReverseV2.h>

#include <loco.h>

namespace luci
{

bool CircleReverseV2GraphBuilder::validate(const ValidateArgs &args) const
{
  if (!GraphBuilder::validate(args, 2))
    return false;

  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;
  const auto tensors = args.reader.tensors();
  const auto &tensor_in = tensors.at(inputs.at(0));
  const auto &tensor_axis = tensors.at(inputs.at(1));
  const auto &tensor_out = tensors.at(outputs[0]);

  switch (tensor_axis->type())
  {
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
      break;
    default:
      return false;
  }

  if (tensor_out->type() != tensor_in->type())
    return false;

  return true;
}

CircleNode *CircleReverseV2GraphBuilder::build_node(const circle::OperatorT &,
                                                    const std::vector<CircleNode *> &inputs,
                                                    loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleReverseV2>();
  node->tensor(inputs.at(0));
  node->axis(inputs.at(1));

  return node;
}

} // namespace luci
