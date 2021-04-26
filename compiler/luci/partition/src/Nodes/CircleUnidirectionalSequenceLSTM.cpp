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

#include "ConnectNode.h"

namespace
{

void connect(luci::ConnectNode *cn, const luci::CircleUnidirectionalSequenceLSTM *node)
{
  auto *cloned = loco::must_cast<luci::CircleUnidirectionalSequenceLSTM *>(cn->find_clone(node));

  luci::CircleNode *input = loco::must_cast<luci::CircleNode *>(node->input());

  luci::CircleNode *input_to_input_weights =
    loco::must_cast<luci::CircleNode *>(node->input_to_input_weights());
  luci::CircleNode *input_to_forget_weights =
    loco::must_cast<luci::CircleNode *>(node->input_to_forget_weights());
  luci::CircleNode *input_to_cell_weights =
    loco::must_cast<luci::CircleNode *>(node->input_to_cell_weights());
  luci::CircleNode *input_to_output_weights =
    loco::must_cast<luci::CircleNode *>(node->input_to_output_weights());

  luci::CircleNode *recurrent_to_input_weights =
    loco::must_cast<luci::CircleNode *>(node->recurrent_to_input_weights());
  luci::CircleNode *recurrent_to_forget_weights =
    loco::must_cast<luci::CircleNode *>(node->recurrent_to_forget_weights());
  luci::CircleNode *recurrent_to_cell_weights =
    loco::must_cast<luci::CircleNode *>(node->recurrent_to_cell_weights());
  luci::CircleNode *recurrent_to_output_weights =
    loco::must_cast<luci::CircleNode *>(node->recurrent_to_output_weights());

  luci::CircleNode *cell_to_input_weights =
    loco::must_cast<luci::CircleNode *>(node->cell_to_input_weights());
  luci::CircleNode *cell_to_forget_weights =
    loco::must_cast<luci::CircleNode *>(node->cell_to_forget_weights());
  luci::CircleNode *cell_to_output_weights =
    loco::must_cast<luci::CircleNode *>(node->cell_to_output_weights());

  luci::CircleNode *input_gate_bias = loco::must_cast<luci::CircleNode *>(node->input_gate_bias());
  luci::CircleNode *forget_gate_bias =
    loco::must_cast<luci::CircleNode *>(node->forget_gate_bias());
  luci::CircleNode *cell_gate_bias = loco::must_cast<luci::CircleNode *>(node->cell_gate_bias());
  luci::CircleNode *output_gate_bias =
    loco::must_cast<luci::CircleNode *>(node->output_gate_bias());

  luci::CircleNode *projection_weights =
    loco::must_cast<luci::CircleNode *>(node->projection_weights());
  luci::CircleNode *projection_bias = loco::must_cast<luci::CircleNode *>(node->projection_bias());

  luci::CircleNode *activation_state =
    loco::must_cast<luci::CircleNode *>(node->activation_state());
  luci::CircleNode *cell_state = loco::must_cast<luci::CircleNode *>(node->cell_state());

  luci::CircleNode *input_layer_norm_coefficients =
    loco::must_cast<luci::CircleNode *>(node->input_layer_norm_coefficients());
  luci::CircleNode *forget_layer_norm_coefficients =
    loco::must_cast<luci::CircleNode *>(node->forget_layer_norm_coefficients());
  luci::CircleNode *cell_layer_norm_coefficients =
    loco::must_cast<luci::CircleNode *>(node->cell_layer_norm_coefficients());
  luci::CircleNode *output_layer_norm_coefficients =
    loco::must_cast<luci::CircleNode *>(node->output_layer_norm_coefficients());

  cloned->input(cn->find_clone(input));

  cloned->input_to_input_weights(cn->find_clone(input_to_input_weights));
  cloned->input_to_forget_weights(cn->find_clone(input_to_forget_weights));
  cloned->input_to_cell_weights(cn->find_clone(input_to_cell_weights));
  cloned->input_to_output_weights(cn->find_clone(input_to_output_weights));

  cloned->recurrent_to_input_weights(cn->find_clone(recurrent_to_input_weights));
  cloned->recurrent_to_forget_weights(cn->find_clone(recurrent_to_forget_weights));
  cloned->recurrent_to_cell_weights(cn->find_clone(recurrent_to_cell_weights));
  cloned->recurrent_to_output_weights(cn->find_clone(recurrent_to_output_weights));

  cloned->cell_to_input_weights(cn->find_clone(cell_to_input_weights));
  cloned->cell_to_forget_weights(cn->find_clone(cell_to_forget_weights));
  cloned->cell_to_output_weights(cn->find_clone(cell_to_output_weights));

  cloned->input_gate_bias(cn->find_clone(input_gate_bias));
  cloned->forget_gate_bias(cn->find_clone(forget_gate_bias));
  cloned->cell_gate_bias(cn->find_clone(cell_gate_bias));
  cloned->output_gate_bias(cn->find_clone(output_gate_bias));

  cloned->projection_weights(cn->find_clone(projection_weights));
  cloned->projection_bias(cn->find_clone(projection_bias));

  cloned->activation_state(cn->find_clone(activation_state));
  cloned->cell_state(cn->find_clone(cell_state));

  cloned->input_layer_norm_coefficients(cn->find_clone(input_layer_norm_coefficients));
  cloned->forget_layer_norm_coefficients(cn->find_clone(forget_layer_norm_coefficients));
  cloned->cell_layer_norm_coefficients(cn->find_clone(cell_layer_norm_coefficients));
  cloned->output_layer_norm_coefficients(cn->find_clone(output_layer_norm_coefficients));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleUnidirectionalSequenceLSTM *node) { connect(this, node); }

} // namespace luci
