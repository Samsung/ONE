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

#include "luci/Import/Nodes/CircleBidirectionalSequenceLSTM.h"

#include <luci/IR/Nodes/CircleBidirectionalSequenceLSTM.h>
#include <luci/IR/Nodes/CircleBidirectionalSequenceLSTMOut.h>

#include <loco.h>

namespace luci
{

bool CircleBidirectionalSequenceLSTMGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 48)
    return false;

  return true;
}

CircleNode *CircleBidirectionalSequenceLSTMGraphBuilder::build_node(
  const circle::OperatorT &op, const std::vector<CircleNode *> &inputs, loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleBidirectionalSequenceLSTM>();
  node->input(inputs.at(0));
  node->fw_input_to_input_weights(inputs.at(1)); // Optional
  node->fw_input_to_cell_weights(inputs.at(2));
  node->fw_input_to_forget_weights(inputs.at(3));
  node->fw_input_to_output_weights(inputs.at(4));
  node->fw_recurrent_to_input_weights(inputs.at(5)); // Optional
  node->fw_recurrent_to_cell_weights(inputs.at(6));
  node->fw_recurrent_to_forget_weights(inputs.at(7));
  node->fw_recurrent_to_output_weights(inputs.at(8));
  node->fw_cell_to_input_weights(inputs.at(9));   // Optional
  node->fw_cell_to_forget_weights(inputs.at(10)); // Optional
  node->fw_cell_to_output_weights(inputs.at(11)); // Optional
  node->fw_input_gate_bias(inputs.at(12));        // Optional
  node->fw_forget_gate_bias(inputs.at(13));
  node->fw_cell_gate_bias(inputs.at(14));
  node->fw_output_gate_bias(inputs.at(15));
  node->fw_projection_weights(inputs.at(16));     // Optional
  node->fw_projection_bias(inputs.at(17));        // Optional
  node->bw_input_to_input_weights(inputs.at(18)); // Optional
  node->bw_input_to_cell_weights(inputs.at(19));
  node->bw_input_to_forget_weights(inputs.at(20));
  node->bw_input_to_output_weights(inputs.at(21));
  node->bw_recurrent_to_input_weights(inputs.at(22)); // Optional
  node->bw_recurrent_to_cell_weights(inputs.at(23));
  node->bw_recurrent_to_forget_weights(inputs.at(24));
  node->bw_recurrent_to_output_weights(inputs.at(25));
  node->bw_cell_to_input_weights(inputs.at(26));  // Optional
  node->bw_cell_to_forget_weights(inputs.at(27)); // Optional
  node->bw_cell_to_output_weights(inputs.at(28)); // Optional
  node->bw_input_gate_bias(inputs.at(29));        // Optional
  node->bw_forget_gate_bias(inputs.at(30));
  node->bw_cell_gate_bias(inputs.at(31));
  node->bw_output_gate_bias(inputs.at(32));
  node->bw_projection_weights(inputs.at(33)); // Optional
  node->bw_projection_bias(inputs.at(34));    // Optional
  node->fw_activation_state(inputs.at(35));
  node->fw_cell_state(inputs.at(36));
  node->bw_activation_state(inputs.at(37));
  node->bw_cell_state(inputs.at(38));

  node->auxillary_input(inputs.at(39));                      // Optional
  node->fw_auxillary_input_to_input_weights(inputs.at(40));  // Optional
  node->fw_auxillary_input_to_forget_weights(inputs.at(41)); // Optional
  node->fw_auxillary_input_to_cell_weights(inputs.at(42));   // Optional
  node->fw_auxillary_input_to_output_weights(inputs.at(43)); // Optional
  node->bw_auxillary_input_to_input_weights(inputs.at(44));  // Optional
  node->bw_auxillary_input_to_forget_weights(inputs.at(45)); // Optional
  node->bw_auxillary_input_to_cell_weights(inputs.at(46));   // Optional
  node->bw_auxillary_input_to_output_weights(inputs.at(47)); // Optional

  const auto *options = op.builtin_options.AsBidirectionalSequenceLSTMOptions();
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));
  node->cell_clip(options->cell_clip);
  node->proj_clip(options->proj_clip);
  node->merge_outputs(options->merge_outputs);
  node->time_major(options->time_major);
  node->asymmetric_quantize_inputs(options->asymmetric_quantize_inputs);

  return node;
}

CircleNode *CircleBidirectionalSequenceLSTMGraphBuilder::build(const circle::OperatorT &op,
                                                               GraphBuilderContext *context) const
{
  assert(context != nullptr);

  const std::vector<int32_t> &inputs = op.inputs;
  const std::vector<int32_t> &outputs = op.outputs;
  const auto &tensors = context->reader()->tensors();
  auto tensors_ptr = context->reader()->tensors_ptr();
  assert(tensors_ptr != nullptr);

  std::vector<CircleNode *> input_nodes;
  for (const int32_t input_tensor_index : inputs)
  {
    if (input_tensor_index >= 0)
    {
      auto input = context->nodefinder()->node(input_tensor_index);
      if (input != nullptr)
        input_nodes.push_back(input);
    }
    else
    {
      // If there is no tensor, insert CircleOutputExclude.
      input_nodes.push_back(context->graph()->nodes()->create<luci::CircleOutputExclude>());
    }
  }

  // Create CircleBidirectionalSequenceLSTM
  CircleNode *node =
    CircleBidirectionalSequenceLSTMGraphBuilder::build_node(op, input_nodes, context->graph());

  assert(int32_t(outputs.size()) == 2);
  // Let's use name of output 0 as BidirectionalSequenceLSTM name
  const circle::TensorT &output_tensor = *tensors[outputs[0]];
  node->name(tensor_name(output_tensor));

  // Create virtual outputs of BidirectionalSequenceLSTM
  for (int32_t n = 0; n < 2; ++n)
  {
    const circle::TensorT &output_tensor = *tensors[outputs[n]];

    auto *nodeout = context->graph()->nodes()->create<CircleBidirectionalSequenceLSTMOut>();
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

  return node;
}

} // namespace luci
