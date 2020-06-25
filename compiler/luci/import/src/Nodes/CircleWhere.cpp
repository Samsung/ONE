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

#include "luci/Import/Nodes/CircleWhere.h"

#include <luci/IR/Nodes/CircleWhere.h>

#include <loco.h>

namespace luci
{

bool CircleWhereGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  auto in_size = inputs.size();

  if ((in_size != 1) && (in_size != 3))
    return false;

  if (outputs.size() != 1)
    return false;

  const auto &tensors = args.reader.tensors();
  const auto &tensor_0 = tensors.at(inputs[0]);
  const auto &tensor_out = tensors.at(outputs[0]);

  if (tensor_0->type != circle::TensorType_BOOL)
    return false;

  if (in_size == 1)
  {
    if (tensor_out->type != circle::TensorType_INT64)
      return false;
  }

  if (in_size == 3)
  {
    const auto &tensor_x = tensors.at(inputs[1]);
    const auto &tensor_y = tensors.at(inputs[2]);

    if (tensor_x->type != tensor_y->type)
      return false;

    if (tensor_out->type != tensor_x->type)
      return false;
  }

  return true;
}

CircleNode *CircleWhereGraphBuilder::build_node(const circle::OperatorT &,
                                                const std::vector<CircleNode *> &inputs,
                                                loco::Graph *graph) const
{
  bool has_xy_inputs = (inputs.size() == 3);
  auto *node = graph->nodes()->create<CircleWhere>(has_xy_inputs);
  node->cond(inputs[0]);
  if (has_xy_inputs)
  {
    node->x(inputs[1]);
    node->y(inputs[2]);
  }

  return node;
}

} // namespace luci
