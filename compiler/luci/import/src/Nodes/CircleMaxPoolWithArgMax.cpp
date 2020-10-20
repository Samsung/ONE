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

#include "luci/Import/Nodes/CircleMaxPoolWithArgMax.h"

#include <luci/IR/Nodes/CircleMaxPoolWithArgMax.h>

#include <loco.h>

namespace luci
{

bool CircleMaxPoolWithArgMaxGraphBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;
  const auto &outputs = args.op.outputs;

  const auto *options = args.op.builtin_options.AsMaxPoolWithArgMaxOptions();

  if (inputs.size() != 1)
    return false;

  if (outputs.size() != 2)
    return false;

  const auto &tensors = args.reader.tensors();
  if (tensors.at(inputs.at(0))->type != tensors.at(outputs[0])->type)
    return false;

  if (options->output_type != tensors.at(outputs[1])->type)
    return false;

  return true;
}

void CircleMaxPoolWithArgMaxGraphBuilder::build(const circle::OperatorT &op,
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
  auto *node = graph->nodes()->create<CircleMaxPoolWithArgMax>();

  node->input(input_nodes[0]);

  const auto *options = op.builtin_options.AsMaxPoolWithArgMaxOptions();

  node->padding(luci_padding(options->padding));
  node->stride()->w(options->stride_w);
  node->stride()->h(options->stride_h);
  node->filter()->w(options->filter_width);
  node->filter()->h(options->filter_height);
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));
  node->output_type(luci_datatype(options->output_type));

  assert(outputs.size() == 2);
  {
    // Let's use name of output 0 as MaxPoolWithArgMax name
    const circle::TensorT &output_tensor = *tensors[outputs[0]];
    node->name(tensor_name(output_tensor));
    node->op_version(opcodes[op.opcode_index].get()->version);

    // NOTE We don't set quantization for MaxPoolWithArgMax itself but to virtual outputs
  }

  // Create virtual outputs of NonMaxSuppressionV4
  for (size_t n = 0; n < outputs.size(); ++n)
  {
    const circle::TensorT &output_tensor = *tensors[outputs[n]];

    auto *nodeout = graph->nodes()->create<CircleMaxPoolWithArgMaxOut>();
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
