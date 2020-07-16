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

#include "luci/Import/Nodes/CircleSelectV2.h"

#include <luci/IR/Nodes/CircleSelectV2.h>

#include <loco.h>

namespace luci
{

bool CircleSelectV2GraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;
  if (inputs.size() != 3)
    return false;
  if (outputs.size() != 1)
    return false;

  const auto &tensors = args.reader.tensors();
  const auto &condition = tensors.at(inputs[0]);
  if (condition->type != circle::TensorType_BOOL)
    return false;

  const auto &t = tensors.at(inputs[1]);
  const auto &e = tensors.at(inputs[2]);
  if (t->type != e->type)
    return false;

  return true;
}

CircleNode *CircleSelectV2GraphBuilder::build_node(const circle::OperatorT &,
                                                   const std::vector<CircleNode *> &inputs,
                                                   loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleSelectV2>();
  node->condition(inputs[0]);
  node->t(inputs[1]);
  node->e(inputs[2]);

  return node;
}

} // namespace luci
