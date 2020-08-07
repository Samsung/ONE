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

#include "luci/Import/Nodes/CircleGatherNd.h"

#include <luci/IR/Nodes/CircleGatherNd.h>

#include <loco.h>
#include <oops/UserExn.h>
#include <mio/circle/schema_generated.h>

namespace luci
{

bool CircleGatherNdGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  if (inputs.size() != 2)
    return false;

  if (outputs.size() != 1)
    return false;

  auto &indices_tensor = args.reader.tensors()[inputs.at(1)];

  if (!(indices_tensor->type == circle::TensorType::TensorType_INT32 ||
        indices_tensor->type == circle::TensorType::TensorType_INT64))
  {
    return false;
  }

  return true;
}

CircleNode *CircleGatherNdGraphBuilder::build_node(const circle::OperatorT &,
                                                   const std::vector<CircleNode *> &inputs,
                                                   loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleGatherNd>();

  node->params(inputs.at(0));
  node->indices(inputs.at(1));

  // GatherNd options empty

  return node;
}

} // namespace luci
