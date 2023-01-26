/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
build_kernel_CircleUnidirectionalSequenceLSTM(std::vector<const Tensor *> &&inputs,
                                              std::vector<Tensor *> &&outputs,
                                              const uint32_t op_index, KernelBuilder &builder)
{
  assert(inputs.size() == 24);
  const Tensor *input = inputs.at(0);
  const Tensor *input_to_input_weights = inputs.at(1);
  const Tensor *input_to_forget_weights = inputs.at(2);
  const Tensor *input_to_cell_weights = inputs.at(3);
  const Tensor *input_to_output_weights = inputs.at(4);

  const Tensor *recurrent_to_input_weights = inputs.at(5);
  const Tensor *recurrent_to_forget_weights = inputs.at(6);
  const Tensor *recurrent_to_cell_weights = inputs.at(7);
  const Tensor *recurrent_to_output_weights = inputs.at(8);

  const Tensor *cell_to_input_weights = inputs.at(9);
  const Tensor *cell_to_forget_weights = inputs.at(10);
  const Tensor *cell_to_output_weights = inputs.at(11);

  const Tensor *input_gate_bias = inputs.at(12);
  const Tensor *forget_gate_bias = inputs.at(13);
  const Tensor *cell_gate_bias = inputs.at(14);
  const Tensor *output_gate_bias = inputs.at(15);

  const Tensor *projection_weights = inputs.at(16);
  const Tensor *projection_bias = inputs.at(17);

  Tensor *output_state = const_cast<Tensor *>(inputs.at(18));
  Tensor *cell_state = const_cast<Tensor *>(inputs.at(19));

  const Tensor *input_layer_norm_coefficients = inputs.at(20);
  const Tensor *forget_layer_norm_coefficients = inputs.at(21);
  const Tensor *cell_layer_norm_coefficients = inputs.at(22);
  const Tensor *output_layer_norm_coefficients = inputs.at(23);

  circle::OperatorT oper_t;
  builder.get_circle_reader()->operators()[op_index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsUnidirectionalSequenceLSTMOptions();

  UnidirectionalSequenceLSTMParams params{};
  params.activation = luci_actfunc(options->fused_activation_function);
  params.cell_clip = options->cell_clip;
  params.proj_clip = options->proj_clip;
  params.time_major = options->time_major;
  params.asymmetric_quantize_inputs = options->asymmetric_quantize_inputs;

  const bool is_integer = input->element_type() == DataType::S8;
  bool use_layer_norm = (forget_layer_norm_coefficients != nullptr);
  if (is_integer)
  {
    if (not use_layer_norm)
    {
      params.intermediate_affine_quant =
        builder.get_runtime_graph()->getIntermediateAffineQuantizations();

      // For integer LSTM need 4 16-bit buffer with size n_batch * n_cell
      // and 1 8-bit buffer with size n_batch * n_cell
      auto tmp_1 = std::make_unique<Tensor>(DataType::S16, Shape({}), nullptr);
      tmp_1->set_data_buffer(nullptr);
      outputs.push_back(builder.get_runtime_graph()->addTensor(std::move(tmp_1)));

      auto tmp_2 = std::make_unique<Tensor>(DataType::S16, Shape({}), nullptr);
      tmp_2->set_data_buffer(nullptr);
      outputs.push_back(builder.get_runtime_graph()->addTensor(std::move(tmp_2)));

      auto tmp_3 = std::make_unique<Tensor>(DataType::S16, Shape({}), nullptr);
      tmp_3->set_data_buffer(nullptr);
      outputs.push_back(builder.get_runtime_graph()->addTensor(std::move(tmp_3)));

      auto tmp_4 = std::make_unique<Tensor>(DataType::S16, Shape({}), nullptr);
      tmp_4->set_data_buffer(nullptr);
      outputs.push_back(builder.get_runtime_graph()->addTensor(std::move(tmp_4)));

      auto tmp_5 = std::make_unique<Tensor>(
        DataType::S8, Shape({}),
        builder.get_runtime_graph()->getIntermediateAffineQuantizations()[0]);
      tmp_5->set_data_buffer(nullptr);
      outputs.push_back(builder.get_runtime_graph()->addTensor(std::move(tmp_5)));
    }
    else
    {
      // TODO: support float
      assert(false && "Not supported now");
    }
  }

  outputs.push_back(output_state);
  outputs.push_back(cell_state);

  return std::make_unique<kernels::UnidirectionalSequenceLSTM>(
    input, input_to_input_weights, input_to_forget_weights, input_to_cell_weights,
    input_to_output_weights, recurrent_to_input_weights, recurrent_to_forget_weights,
    recurrent_to_cell_weights, recurrent_to_output_weights, cell_to_input_weights,
    cell_to_forget_weights, cell_to_output_weights, input_gate_bias, forget_gate_bias,
    cell_gate_bias, output_gate_bias, projection_weights, projection_bias,
    input_layer_norm_coefficients, forget_layer_norm_coefficients, cell_layer_norm_coefficients,
    output_layer_norm_coefficients, std::move(outputs), params);
}

} // namespace luci_interpreter
