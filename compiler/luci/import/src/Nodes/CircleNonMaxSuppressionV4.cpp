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

#include "luci/Import/Nodes/CircleNonMaxSuppressionV4.h"

#include <luci/IR/Nodes/CircleNonMaxSuppressionV4.h>
#include <luci/IR/Nodes/CircleNonMaxSuppressionV4Out.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleNonMaxSuppressionV4GraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  if (inputs.size() != 5)
    return false;
  if (outputs.size() != 2)
    return false;

  const auto &tensors = args.reader.tensors();
  const auto &boxes_tensor = tensors.at(inputs[0]);
  if (boxes_tensor->shape.size() != 2)
    return false;
  if (boxes_tensor->shape.at(1) != 4)
    return false;
  if (boxes_tensor->shape.at(0) != tensors.at(inputs[1])->shape.at(0))
    return false;

  if (tensors.at(inputs[2])->type != circle::TensorType_INT32)
    return false;
  if (tensors.at(inputs[3])->type != circle::TensorType_FLOAT32)
    return false;
  if (tensors.at(inputs[4])->type != circle::TensorType_FLOAT32)
    return false;

  return true;
}

/**
 * @brief  NonMaxSuppressionV4 Node builder
 *
 * @note   Current loco does not provide multiple outputs
 *         We will create multiple NonMasSuppressionV4Oout nodes to emulate this
 */

void CircleNonMaxSuppressionV4GraphBuilder::build(const circle::OperatorT &op,
                                                  GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();
  const auto &opcodes = context->reader()->opcodes();
  auto tensors_ptr = context->reader()->tensors_ptr();
  assert(tensors_ptr != nullptr);

  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    input_nodes.push_back(context->nodefinder()->node(input_tensor_index));
  }

  // Create CircleNonMaxSuppressionV4
  auto node = graph->nodes()->create<CircleNonMaxSuppressionV4>();
  node->boxes(input_nodes[0]);
  node->scores(input_nodes[1]);
  node->max_output_size(input_nodes[2]);
  node->iou_threshold(input_nodes[3]);
  node->score_threshold(input_nodes[4]);

  assert(outputs.size() == 2);
  {
    // Let's use name of output 0 as NonMaxSuppressionV4 name
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    node->name(tensor_name(output_tensor));
    node->op_version(opcodes[op.opcode_index].get()->version);

    // NOTE We don't set quantization for NonMaxSuppressionV4 itself but to virtual outputs
  }

  // Create virtual outputs of NonMaxSuppressionV4
  for (int32_t n = 0; n < int32_t(outputs.size()); ++n)
  {
    const circle::TensorT &output_tensor = *tensors[outputs[n]];

    auto *nodeout = graph->nodes()->create<CircleNonMaxSuppressionV4Out>();
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
