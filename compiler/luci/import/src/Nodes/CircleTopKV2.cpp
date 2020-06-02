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

#include "luci/Import/Nodes/CircleTopKV2.h"

#include <luci/IR/Nodes/CircleTopKV2.h>
#include <luci/IR/Nodes/CircleTopKV2Out.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleTopKV2GraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  if (inputs.size() != 2)
    return false;
  if (outputs.size() != 2)
    return false;

  const auto &tensors = args.reader.tensors();
  const auto &tensor = tensors.at(inputs[1]);
  if (tensor->type != circle::TensorType_INT32)
    return false;

  return true;
}

/**
 * @brief  TopKV2 Node builder
 *
 * @note   Current loco does not provide multiple outputs
 *         We will create multiple CircleTopKV2Out nodes to emulate this
 *         For two outputs that may look like this
 *
 *         --- CircleTopKV2--- FullyConnected ---
 *                           \- FullyConnected ---
 *
 *         will be created like this
 *
 *         --- CircleTopKV2 --- CircleTopKV2Out --- FullyConnected ---
 *                           \- CircleTopKV2Out --- FullyConnected ---
 */

void CircleTopKV2GraphBuilder::build(const circle::OperatorT &op,
                                     GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();
  auto tensors_ptr = context->reader()->tensors_ptr();
  assert(tensors_ptr != nullptr);

  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    input_nodes.push_back(context->nodefinder()->node(input_tensor_index));
  }

  // Create CircleTopKV2
  auto node = graph->nodes()->create<CircleTopKV2>();
  node->input(input_nodes[0]);
  node->k(input_nodes[1]);

  assert(outputs.size() == 2);
  {
    // Let's use name of output 0 as TopKV2 name
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    node->name(tensor_name(output_tensor));

    // NOTE We don't set quantization for TopKV2 itself but to virtual outputs
  }

  // Create virtual outputs of TopKV2
  for (size_t n = 0; n < outputs.size(); ++n)
  {
    const circle::TensorT &output_tensor = *tensors[outputs[n]];

    auto *nodeout = graph->nodes()->create<CircleTopKV2Out>();
    copy_tensor_attributes(output_tensor, nodeout);
    // mark shape_status
    if (tensors_ptr->Get(outputs[n])->shape() == nullptr)
      nodeout->shape_status(ShapeStatus::NOSHAPE);
    else
      nodeout->shape_status(ShapeStatus::VALID);

    nodeout->input(node);
    nodeout->index(n);

    context->nodefinder()->enroll(outputs[n], nodeout);
  }
}

} // namespace luci
