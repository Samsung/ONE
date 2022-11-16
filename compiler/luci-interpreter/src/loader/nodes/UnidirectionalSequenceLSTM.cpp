/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "Builders.h"

#include "kernels/UnidirectionalSequenceLSTM.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel>
build_kernel_CircleUnidirectionalSequenceLSTM(const luci::CircleNode *circle_node,
                                              KernelBuilderHelper &helper)
{
  const auto *node = loco::must_cast<const luci::CircleUnidirectionalSequenceLSTM *>(circle_node);
  assert(node->arity() == 24);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *input_to_input_weights =
    helper.getOptionalInputTensor(node->input_to_input_weights());
  const Tensor *input_to_cell_weights = helper.getInputTensor(node->input_to_cell_weights());
  const Tensor *input_to_forget_weights = helper.getInputTensor(node->input_to_forget_weights());
  const Tensor *input_to_output_weights = helper.getInputTensor(node->input_to_output_weights());
  const Tensor *recurrent_to_input_weights =
    helper.getOptionalInputTensor(node->recurrent_to_input_weights());
  const Tensor *recurrent_to_cell_weights =
    helper.getInputTensor(node->recurrent_to_cell_weights());
  const Tensor *recurrent_to_forget_weights =
    helper.getInputTensor(node->recurrent_to_forget_weights());
  const Tensor *recurrent_to_output_weights =
    helper.getInputTensor(node->recurrent_to_output_weights());
  const Tensor *cell_to_input_weights =
    helper.getOptionalInputTensor(node->cell_to_input_weights());
  const Tensor *cell_to_forget_weights =
    helper.getOptionalInputTensor(node->cell_to_forget_weights());
  const Tensor *cell_to_output_weights =
    helper.getOptionalInputTensor(node->cell_to_output_weights());
  const Tensor *input_gate_bias = helper.getOptionalInputTensor(node->input_gate_bias());
  const Tensor *forget_gate_bias = helper.getInputTensor(node->forget_gate_bias());
  const Tensor *cell_gate_bias = helper.getInputTensor(node->cell_gate_bias());
  const Tensor *output_gate_bias = helper.getInputTensor(node->output_gate_bias());
  const Tensor *projection_weights = helper.getOptionalInputTensor(node->projection_weights());
  const Tensor *projection_bias = helper.getOptionalInputTensor(node->projection_bias());
  const Tensor *output_state = helper.getInputTensor(node->output_state());
  const Tensor *cell_state = helper.getInputTensor(node->cell_state());
  const Tensor *input_layer_norm_coefficients =
    helper.getOptionalInputTensor(node->input_layer_norm_coefficients());
  const Tensor *forget_layer_norm_coefficients =
    helper.getOptionalInputTensor(node->forget_layer_norm_coefficients());
  const Tensor *cell_layer_norm_coefficients =
    helper.getOptionalInputTensor(node->cell_layer_norm_coefficients());
  const Tensor *output_layer_norm_coefficients =
    helper.getOptionalInputTensor(node->output_layer_norm_coefficients());
  Tensor *output = helper.getOutputTensor(node);

  // scratch pad tensor
  // NOTE provide more scratch pads if support hybrid or integer
  auto sp_output_state =
    std::make_unique<Tensor>(output_state->element_type(), Shape({}), AffineQuantization{}, "");
  sp_output_state->set_observable(false);
  sp_output_state->set_data_buffer(nullptr);
  Tensor *tmp_1 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(sp_output_state));

  auto sp_cell_state =
    std::make_unique<Tensor>(cell_state->element_type(), Shape({}), AffineQuantization{}, "");
  sp_cell_state->set_observable(false);
  sp_cell_state->set_data_buffer(nullptr);
  Tensor *tmp_2 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(sp_cell_state));

  auto sp_3 = std::make_unique<Tensor>(input->element_type(), Shape({}), AffineQuantization{}, "");
  sp_3->set_observable(false);
  sp_3->set_data_buffer(nullptr);
  Tensor *tmp_3 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(sp_3));

  UnidirectionalSequenceLSTMParams params{};
  params.activation = node->fusedActivationFunction();
  params.cell_clip = node->cell_clip();
  params.proj_clip = node->proj_clip();
  params.time_major = node->time_major();
  params.asymmetric_quantize_inputs = node->asymmetric_quantize_inputs();

  return std::make_unique<kernels::UnidirectionalSequenceLSTM>(
    input, input_to_input_weights, input_to_forget_weights, input_to_cell_weights,
    input_to_output_weights, recurrent_to_input_weights, recurrent_to_forget_weights,
    recurrent_to_cell_weights, recurrent_to_output_weights, cell_to_input_weights,
    cell_to_forget_weights, cell_to_output_weights, input_gate_bias, forget_gate_bias,
    cell_gate_bias, output_gate_bias, projection_weights, projection_bias, output_state, cell_state,
    input_layer_norm_coefficients, forget_layer_norm_coefficients, cell_layer_norm_coefficients,
    output_layer_norm_coefficients, output, tmp_1, tmp_2, tmp_3, params);
}

} // namespace luci_interpreter
