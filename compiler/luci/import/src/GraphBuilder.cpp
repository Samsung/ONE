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

#include <luci/IR/ProfilingData.h>
#include <luci/Log.h>

namespace luci
{

void GraphBuilder::build(const circle::OperatorT &op, GraphBuilderContext *context) const
{
  LOGGER(l);

  assert(context != nullptr);

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();
  const auto &opcodes = context->reader()->opcodes();
  const auto &operators = context->reader()->operators();
  auto tensors_ptr = context->reader()->tensors_ptr();
  assert(tensors_ptr != nullptr);

  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    if (input_tensor_index >= 0)
    {
      auto input = context->nodefinder()->node(input_tensor_index);
      if (input == nullptr)
        INFO(l) << "[luci] Warning: input node is null " << input_tensor_index << std::endl;
      input_nodes.push_back(input);
    }
    else
    {
      // If there is no tensor, insert CircleOutputExclude.
      input_nodes.push_back(context->graph()->nodes()->create<luci::CircleOutputExclude>());
    }
  }

  CircleNode *node = build_node(op, input_nodes, context->graph());

  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    const circle::OperatorT &ref_op = *operators[i];
    if (&ref_op == &op)
    {
      node->annot(std::make_unique<luci::CircleNodeID>(i));
      node->annot(std::make_unique<luci::CircleNodeOrigin>(i));
      break;
    }
  }

  // Set up node parameters.
  assert(outputs.size() == 1);
  {
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    copy_tensor_attributes(output_tensor, node);
    // mark shape_status
    if (tensors_ptr->Get(outputs[0])->shape() == nullptr)
      node->shape_status(ShapeStatus::NOSHAPE);
    else
      node->shape_status(ShapeStatus::VALID);

    // mark operator version
    node->op_version(opcodes[op.opcode_index].get()->version);
  }

  // Register node's only output.
  assert(outputs.size() == 1);
  {
    context->nodefinder()->enroll(outputs[0], node);
  }
}

} // namespace luci
