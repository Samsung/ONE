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

#include "luci/Import/Nodes/CircleSquaredDifference.h"

#include <luci/IR/Nodes/CircleSquaredDifference.h>

#include <loco.h>

namespace luci
{

bool CircleSquaredDifferenceGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  if (inputs.size() != 2)
    return false;

  if (outputs.size() != 1)
    return false;

  // Inputs must be one of the following types
  // bfloat16, half(float16), float32, float64, int32, int64, complex64, complex128
  const auto &tensors = args.reader.tensors();
  const auto &tensor = tensors.at(inputs[0]);
  switch (tensor->type)
  {
    case circle::TensorType_FLOAT16:
    case circle::TensorType_FLOAT32:
    case circle::TensorType_FLOAT64:
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
    case circle::TensorType_COMPLEX64:
      break;
    // TODO support bfloat16, complex128
    default:
      return false;
  }

  // Input types must match
  if (tensors.at(inputs[0])->type != tensors.at(inputs[1])->type)
    return false;

  // Input and output types must match
  if (tensors.at(inputs[0])->type != tensors.at(outputs[0])->type)
    return false;

  return true;
}

CircleNode *CircleSquaredDifferenceGraphBuilder::build_node(const circle::OperatorT &,
                                                            const std::vector<CircleNode *> &inputs,
                                                            loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleSquaredDifference>();
  node->x(inputs[0]);
  node->y(inputs[1]);

  return node;
}

} // namespace luci
