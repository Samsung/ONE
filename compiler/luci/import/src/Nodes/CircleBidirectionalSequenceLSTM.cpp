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
  if (args.op.outputs.size() != 2)
    return false;

  return true;
}

CircleNode *CircleBidirectionalSequenceLSTMGraphBuilder::build_node(const BuildNodeArgs &bna) const
{
  auto *node = bna.context->graph()->nodes()->create<CircleBidirectionalSequenceLSTM>();
  auto &inputs = bna.input_nodes;
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

  const auto *options = bna.op.builtin_options.AsBidirectionalSequenceLSTMOptions();
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));
  node->cell_clip(options->cell_clip);
  node->proj_clip(options->proj_clip);
  node->merge_outputs(options->merge_outputs);
  node->time_major(options->time_major);
  node->asymmetric_quantize_inputs(options->asymmetric_quantize_inputs);

  return node;
}

CircleNode *CircleBidirectionalSequenceLSTMGraphBuilder::build_out(const BuildOutArgs &boa) const
{
  auto *nodeout = boa.node->graph()->nodes()->create<CircleBidirectionalSequenceLSTMOut>();

  nodeout->input(boa.node);
  nodeout->index(boa.index);

  return nodeout;
}

} // namespace luci
