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

#include "luci/Import/GraphBuilder.h"

namespace luci
{

void GraphBuilder::build(const circle::OperatorT &op, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();

  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    if (input_tensor_index == -1)
    {
      // If it is an optional input, we must specify it as a nullptr or an actual value to see which
      // option is empty when building the actual op. If we continue without nullptr, we will not be
      // able to know which option is empty.
      input_nodes.push_back(nullptr);
    }
    input_nodes.push_back(context->nodefinder()->node(input_tensor_index));
  }

  CircleNode *node = build_node(op, input_nodes, context->graph());

  // Set up node parameters.
  assert(outputs.size() == 1);
  {
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    copy_tensor_attributes(output_tensor, node);
  }

  // Register node's only output.
  assert(outputs.size() == 1);
  {
    context->nodefinder()->enroll(outputs[0], node);
  }
}

} // namespace luci
