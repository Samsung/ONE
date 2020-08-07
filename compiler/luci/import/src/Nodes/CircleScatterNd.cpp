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

#include "luci/Import/Nodes/CircleScatterNd.h"

#include <luci/IR/Nodes/CircleScatterNd.h>

#include <loco.h>

namespace luci
{

bool CircleScatterNdGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  if (inputs.size() != 3)
    return false;

  // indices must have the same type as shape
  const auto &tensors = args.reader.tensors();

  if (tensors[inputs.at(0)]->type != tensors[inputs.at(2)]->type)
    return false;

  // indices must be either int32 or int64
  if (tensors[inputs.at(0)]->type != circle::TensorType_INT32 &&
      tensors[inputs.at(0)]->type != circle::TensorType_INT64)
    return false;

  return true;
}

CircleNode *CircleScatterNdGraphBuilder::build_node(const circle::OperatorT &,
                                                    const std::vector<CircleNode *> &inputs,
                                                    loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleScatterNd>();
  node->indices(inputs.at(0));
  node->updates(inputs.at(1));
  node->shape(inputs.at(2));

  return node;
}

} // namespace luci
