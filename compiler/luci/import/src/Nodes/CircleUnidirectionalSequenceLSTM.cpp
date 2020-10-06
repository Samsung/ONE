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

#include "luci/Import/Nodes/CircleUnidirectionalSequenceLSTM.h"

#include <luci/IR/Nodes/CircleUnidirectionalSequenceLSTM.h>

#include <loco.h>

namespace luci
{

bool CircleUnidirectionalSequenceLSTMGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 24)
    return false;

  return true;
}

CircleNode *CircleUnidirectionalSequenceLSTMGraphBuilder::build_node(
    const circle::OperatorT &op, const std::vector<CircleNode *> &inputs, loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleUnidirectionalSequenceLSTM>();
  node->input(inputs.at(0));

  node->input_to_input_weights(inputs.at(1)); // Optional
  if (auto input_to_input_weights =
          dynamic_cast<luci::CircleOutputExclude *>(node->input_to_input_weights()))
  {
    input_to_input_weights->dtype(loco::DataType::FLOAT32);
  }
  node->input_to_cell_weights(inputs.at(2));
  node->input_to_forget_weights(inputs.at(3));
  node->input_to_output_weights(inputs.at(4));

  node->recurrent_to_input_weights(inputs.at(5)); // Optional
  if (auto recurrent_to_input_weights =
          dynamic_cast<luci::CircleOutputExclude *>(node->recurrent_to_input_weights()))
  {
    recurrent_to_input_weights->dtype(loco::DataType::FLOAT32);
  }
  node->recurrent_to_cell_weights(inputs.at(6));
  node->recurrent_to_forget_weights(inputs.at(7));
  node->recurrent_to_output_weights(inputs.at(8));

  node->cell_to_input_weights(inputs.at(9)); // Optional
  if (auto cell_to_input_weights =
          dynamic_cast<luci::CircleOutputExclude *>(node->cell_to_input_weights()))
  {
    cell_to_input_weights->dtype(loco::DataType::FLOAT32);
  }
  node->cell_to_forget_weights(inputs.at(10)); // Optional
  if (auto cell_to_forget_weights =
          dynamic_cast<luci::CircleOutputExclude *>(node->cell_to_forget_weights()))
  {
    cell_to_forget_weights->dtype(loco::DataType::FLOAT32);
  }
  node->cell_to_output_weights(inputs.at(11)); // Optional
  if (auto cell_to_output_weights =
          dynamic_cast<luci::CircleOutputExclude *>(node->cell_to_output_weights()))
  {
    cell_to_output_weights->dtype(loco::DataType::FLOAT32);
  }

  node->input_gate_bias(inputs.at(12)); // Optional
  if (auto input_gate_bias = dynamic_cast<luci::CircleOutputExclude *>(node->input_gate_bias()))
  {
    input_gate_bias->dtype(loco::DataType::FLOAT32);
  }
  node->forget_gate_bias(inputs.at(13));
  node->cell_gate_bias(inputs.at(14));
  node->output_gate_bias(inputs.at(15));

  node->projection_weights(inputs.at(16)); // Optional
  if (auto projection_weights =
          dynamic_cast<luci::CircleOutputExclude *>(node->projection_weights()))
  {
    projection_weights->dtype(loco::DataType::FLOAT32);
  }
  node->projection_bias(inputs.at(17)); // Optional
  if (auto projection_bias = dynamic_cast<luci::CircleOutputExclude *>(node->projection_bias()))
  {
    projection_bias->dtype(loco::DataType::FLOAT32);
  }

  node->activation_state(inputs.at(18));
  node->cell_state(inputs.at(19));

  node->input_layer_norm_coefficients(inputs.at(20)); // Optional
  if (auto input_layer_norm_coefficients =
          dynamic_cast<luci::CircleOutputExclude *>(node->input_layer_norm_coefficients()))
  {
    input_layer_norm_coefficients->dtype(loco::DataType::FLOAT32);
  }
  node->forget_layer_norm_coefficients(inputs.at(21)); // Optional
  if (auto forget_layer_norm_coefficients =
          dynamic_cast<luci::CircleOutputExclude *>(node->forget_layer_norm_coefficients()))
  {
    forget_layer_norm_coefficients->dtype(loco::DataType::FLOAT32);
  }
  node->cell_layer_norm_coefficients(inputs.at(22)); // Optional
  if (auto cell_layer_norm_coefficients =
          dynamic_cast<luci::CircleOutputExclude *>(node->cell_layer_norm_coefficients()))
  {
    cell_layer_norm_coefficients->dtype(loco::DataType::FLOAT32);
  }
  node->output_layer_norm_coefficients(inputs.at(23)); // Optional
  if (auto output_layer_norm_coefficients =
          dynamic_cast<luci::CircleOutputExclude *>(node->output_layer_norm_coefficients()))
  {
    output_layer_norm_coefficients->dtype(loco::DataType::FLOAT32);
  }

  const auto *options = op.builtin_options.AsUnidirectionalSequenceLSTMOptions();
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));
  node->cell_clip(options->cell_clip);
  node->proj_clip(options->proj_clip);
  node->time_major(options->time_major);

  return node;
}

} // namespace luci
