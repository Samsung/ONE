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

#include "luci/Import/Nodes/CircleSpaceToBatchND.h"

#include <luci/IR/Nodes/CircleSpaceToBatchND.h>

#include <loco.h>

#include <cassert>

namespace luci
{

bool CircleSpaceToBatchNDGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  if (inputs.size() != 3)
    return false;

  // input 1 and 2 should have INT32/INT64 type
  const auto &tensors = args.reader.tensors();
  const auto &tensor_1 = tensors.at(inputs.at(1));
  switch (tensor_1->type)
  {
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
      break;
    default:
      return false;
  }
  const auto &tensor_2 = tensors.at(inputs.at(2));
  switch (tensor_2->type)
  {
    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
      break;
    default:
      return false;
  }

  // Only support input shape dimension 3 and 4 only
  const auto &tensor_0 = tensors.at(inputs.at(0));
  const auto t_0_s = tensor_0->shape.size();
  if (t_0_s != 3 && t_0_s != 4)
    return false;

  // TODO check input shape

  return true;
}

CircleNode *CircleSpaceToBatchNDGraphBuilder::build_node(const circle::OperatorT &,
                                                         const std::vector<CircleNode *> &inputs,
                                                         loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleSpaceToBatchND>();
  node->input(inputs.at(0));
  node->block_shape(inputs.at(1));
  node->paddings(inputs.at(2));

  // No options for SpaceToBatchND

  return node;
}

} // namespace luci
