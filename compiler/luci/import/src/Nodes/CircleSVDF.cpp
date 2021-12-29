/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/Nodes/CircleSVDF.h"

#include <luci/IR/Nodes/CircleSVDF.h>

#include <loco.h>

#include <cassert>

namespace luci
{
bool CircleSVDFBuilder::validate(const ValidateArgs &args) const
{
  const auto &inputs = args.op.inputs;

  if (inputs.size() != args.op.inputs.size())
    return false;

  return true;
}

/**
 * @brief  SVDF Node builder
 *
 * @note
 */

CircleNode *CircleSVDFBuilder::build(const circle::OperatorT &op,
                                     GraphBuilderContext *context) const
{
  assert(context != nullptr);

  auto graph = context->graph();
  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto tensors = context->reader()->tensors();
  const auto opcodes = context->reader()->opcodes();
  int32_t scratch_index;

  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    auto input_node = context->nodefinder()->node(input_tensor_index);
    if (input_node != nullptr)
      input_nodes.push_back(input_node);
    else
      scratch_index = input_tensor_index;
  }

  auto svdf_scratch_tensor = tensors.at(scratch_index);
  auto *svdf_scratch_node = graph->nodes()->create<CircleDanglingNode>();
  copy_tensor_attributes(svdf_scratch_tensor, svdf_scratch_node);

  if (svdf_scratch_tensor->shape() != nullptr)
    svdf_scratch_node->shape_status(luci::ShapeStatus::VALID);
  else
    svdf_scratch_node->shape_status(luci::ShapeStatus::NOSHAPE);

  context->nodefinder()->enroll(inputs[4], svdf_scratch_node);

  auto *node = graph->nodes()->create<CircleSVDF>();
  node->input(input_nodes[0]);
  node->weight_feature(input_nodes[1]);
  node->weight_time(input_nodes[2]);
  node->bias(input_nodes[3]);
  node->input_activation_state(svdf_scratch_node);

  const auto *options = op.builtin_options.AsSVDFOptions();
  node->asymmetric_quantize_inputs(options->asymmetric_quantize_inputs);
  node->svdf_rank(options->rank);
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));

  // Set up node parameters.
  assert(outputs.size() == 1);
  {
    const auto output_tensor = tensors[outputs[0]];
    assert(output_tensor != nullptr);
    copy_tensor_attributes(output_tensor, node);
    // mark shape_status
    if (output_tensor->shape() == nullptr)
      node->shape_status(ShapeStatus::NOSHAPE);
    else
      node->shape_status(ShapeStatus::VALID);

    // mark operator version
    assert(opcodes[op.opcode_index] != nullptr);
    node->op_version(opcodes[op.opcode_index]->version());
  }

  // Register node's only output.
  assert(outputs.size() == 1);
  {
    context->nodefinder()->enroll(outputs[0], node);
  }

  return node;
}

} // namespace luci
