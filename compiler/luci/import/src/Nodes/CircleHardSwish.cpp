/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleHardSwish.h"

#include <luci/IR/Nodes/CircleHardSwish.h>

#include <loco.h>

namespace luci
{

bool CircleHardSwishGraphBuilder::validate(const ValidateArgs &args) const
{
  if (!GraphBuilder::validate(args, 1))
    return false;

  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  const auto tensors = args.reader.tensors();
  const auto tensor = tensors.at(inputs.at(0));
  assert(tensor != nullptr);

  switch (tensor->type())
  {
    case circle::TensorType_FLOAT64:
      break;
    case circle::TensorType_FLOAT32:
      break;
    case circle::TensorType_INT16:
      break;
    case circle::TensorType_UINT8:
      break;
    default:
      return false;
  }

  assert(tensors[outputs[0]] != nullptr);
  if (tensors[outputs[0]]->type() != tensor->type())
    return false;

  return true;
}

CircleNode *CircleHardSwishGraphBuilder::build_node(const circle::OperatorT &,
                                                    const std::vector<CircleNode *> &inputs,
                                                    loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleHardSwish>();
  node->features(inputs.at(0));

  return node;
}

} // namespace luci
