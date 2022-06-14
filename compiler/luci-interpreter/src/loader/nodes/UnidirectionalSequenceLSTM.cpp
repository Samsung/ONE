/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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
namespace
{
void fill_tensor_with_0(Tensor *state_tensor)
{
  switch (state_tensor->element_type())
  {
    case loco::DataType::FLOAT32:
    {
      auto state_data = state_tensor->data<float>();
      std::fill_n(state_data, state_tensor->shape().num_elements(), 0);
      break;
    }
    case loco::DataType::S32:
    {
      auto state_data = state_tensor->data<int32_t>();
      std::fill_n(state_data, state_tensor->shape().num_elements(), 0);
      break;
    }
    case loco::DataType::S16:
    {
      auto state_data = state_tensor->data<int16_t>();
      std::fill_n(state_data, state_tensor->shape().num_elements(), 0);
      break;
    }
    case loco::DataType::S8:
    {
      auto state_data = state_tensor->data<int8_t>();
      std::fill_n(state_data, state_tensor->shape().num_elements(), 0);
      break;
    }
    case loco::DataType::U8:
    {
      auto state_data = state_tensor->data<int8_t>();
      std::fill_n(state_data, state_tensor->shape().num_elements(), 0);
      break;
    }
    default:
      throw std::runtime_error("Unsupported type.");
  }
}
} // namespace

std::unique_ptr<Kernel>
build_kernel_CircleUnidirectionalSequenceLSTM(const luci::CircleNode *circle_node,
                                              KernelBuilderHelper &helper)
{
  const auto *node = loco::must_cast<const luci::CircleUnidirectionalSequenceLSTM *>(circle_node);
  assert(node->arity() == 24);

  // Inputs
  // 0
  auto input = helper.getInputTensor(node->input());
  // 1
  auto input_to_input_weights = helper.getOptionalInputTensor(node->input_to_input_weights());
  // 2
  auto input_to_forget_weights = helper.getInputTensor(node->input_to_forget_weights());
  // 3
  auto input_to_cell_weights = helper.getOptionalInputTensor(node->input_to_cell_weights());
  // 4
  auto input_to_output_weights = helper.getInputTensor(node->input_to_output_weights());
  // 5
  auto recurrent_to_input_weights = helper.getInputTensor(node->recurrent_to_input_weights());
  // 6
  auto recurrent_to_forget_weights = helper.getInputTensor(node->recurrent_to_forget_weights());
  // 7
  auto recurrent_to_cell_weights = helper.getInputTensor(node->recurrent_to_cell_weights());
  // 8
  auto recurrent_to_output_weights = helper.getInputTensor(node->recurrent_to_output_weights());
  // 9
  auto cell_to_input_weights = helper.getOptionalInputTensor(node->cell_to_input_weights());
  // 10
  auto cell_to_forget_weights = helper.getOptionalInputTensor(node->cell_to_forget_weights());
  // 11
  auto cell_to_output_weights = helper.getOptionalInputTensor(node->cell_to_output_weights());
  // 12
  auto input_gate_bias = helper.getOptionalInputTensor(node->input_gate_bias());
  // 13
  auto forget_gate_bias = helper.getInputTensor(node->forget_gate_bias());
  // 14
  auto cell_gate_bias = helper.getInputTensor(node->cell_gate_bias());
  // 15
  auto output_gate_bias = helper.getInputTensor(node->output_gate_bias());
  // 16
  auto projection_weights = helper.getOptionalInputTensor(node->projection_weights());
  // 17
  auto projection_bias = helper.getOptionalInputTensor(node->projection_bias());

  // 18
  // Note: it is expected that activation_state input variable tensor reset to zero
  auto activation_state = helper.getOutputTensor(node->activation_state());
  // Reset to zero value
  if (activation_state)
    fill_tensor_with_0(activation_state);

  // 19
  // Note: it is expected that cell_state input variable tensor reset to zero
  auto cell_state = helper.getOutputTensor(node->cell_state());
  // Reset to zero value
  if (cell_state)
    fill_tensor_with_0(cell_state);

  // 20
  auto input_layer_norm_coefficients =
    helper.getOptionalInputTensor(node->input_layer_norm_coefficients());
  // 21
  auto forget_layer_norm_coefficients =
    helper.getOptionalInputTensor(node->forget_layer_norm_coefficients());
  // 22
  auto cell_layer_norm_coefficients =
    helper.getOptionalInputTensor(node->cell_layer_norm_coefficients());
  // 23
  auto output_layer_norm_coefficients =
    helper.getOptionalInputTensor(node->output_layer_norm_coefficients());

  // output
  auto output = helper.getOutputTensor(node);

  // To store output and scratchpad tensors
  std::vector<Tensor *> outputs;
  outputs.push_back(output);

  // Scratch buffer tensor
  auto scratch_buffer_tensor =
    std::make_unique<Tensor>(input->element_type(), Shape({}), AffineQuantization{}, "");
  scratch_buffer_tensor->set_observable(false);
  scratch_buffer_tensor->set_data_buffer(nullptr);
  Tensor *scratch_buffer =
    helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratch_buffer_tensor));
  outputs.push_back(scratch_buffer);

  // If is Hybrid case
  if ((input_to_output_weights->element_type() == loco::DataType::U8 ||
       input_to_output_weights->element_type() == loco::DataType::S8) &&
      input->element_type() == loco::DataType::FLOAT32)
  {
    // input quantized tensor
    auto input_quantized_tensor = std::make_unique<Tensor>(input_to_output_weights->element_type(),
                                                           Shape({}), AffineQuantization{}, "");
    input_quantized_tensor->set_observable(false);
    input_quantized_tensor->set_data_buffer(nullptr);
    Tensor *input_quantized =
      helper.getRuntimeGraph(node->graph())->addTensor(std::move(input_quantized_tensor));
    outputs.push_back(input_quantized);

    // output state quantized tensor
    auto output_state_quantized_tensor = std::make_unique<Tensor>(
      input_to_output_weights->element_type(), Shape({}), AffineQuantization{}, "");
    output_state_quantized_tensor->set_observable(false);
    output_state_quantized_tensor->set_data_buffer(nullptr);
    Tensor *output_state_quantized =
      helper.getRuntimeGraph(node->graph())->addTensor(std::move(output_state_quantized_tensor));
    outputs.push_back(output_state_quantized);

    // cell state quantized tensor
    auto cell_state_quantized_tensor = std::make_unique<Tensor>(
      input_to_output_weights->element_type(), Shape({}), AffineQuantization{}, "");
    cell_state_quantized_tensor->set_observable(false);
    cell_state_quantized_tensor->set_data_buffer(nullptr);
    Tensor *cell_state_quantized =
      helper.getRuntimeGraph(node->graph())->addTensor(std::move(cell_state_quantized_tensor));
    outputs.push_back(cell_state_quantized);

    // input sf tensor
    auto input_sf_tensor =
      std::make_unique<Tensor>(DataType::FLOAT32, Shape({}), AffineQuantization{}, "");
    input_sf_tensor->set_observable(false);
    input_sf_tensor->set_data_buffer(nullptr);
    Tensor *input_sf = helper.getRuntimeGraph(node->graph())->addTensor(std::move(input_sf_tensor));
    outputs.push_back(input_sf);

    // output state sf tensor
    auto output_state_sf_tensor =
      std::make_unique<Tensor>(DataType::FLOAT32, Shape({}), AffineQuantization{}, "");
    output_state_sf_tensor->set_observable(false);
    output_state_sf_tensor->set_data_buffer(nullptr);
    Tensor *output_state_sf =
      helper.getRuntimeGraph(node->graph())->addTensor(std::move(output_state_sf_tensor));
    outputs.push_back(output_state_sf);

    // prod scaling factors tensor
    auto prod_scaling_factors_tensor =
      std::make_unique<Tensor>(DataType::FLOAT32, Shape({}), AffineQuantization{}, "");
    prod_scaling_factors_tensor->set_observable(false);
    prod_scaling_factors_tensor->set_data_buffer(nullptr);
    Tensor *prod_scaling_factors =
      helper.getRuntimeGraph(node->graph())->addTensor(std::move(prod_scaling_factors_tensor));
    outputs.push_back(prod_scaling_factors);

    // recovered cell weights tensor
    auto recovered_cell_weights_tensor =
      std::make_unique<Tensor>(DataType::FLOAT32, Shape({}), AffineQuantization{}, "");
    recovered_cell_weights_tensor->set_observable(false);
    recovered_cell_weights_tensor->set_data_buffer(nullptr);
    Tensor *recovered_cell_weights =
      helper.getRuntimeGraph(node->graph())->addTensor(std::move(recovered_cell_weights_tensor));
    outputs.push_back(recovered_cell_weights);

    // accum scratch tensor
    auto accum_scratch_tensor =
      std::make_unique<Tensor>(DataType::S32, Shape({}), AffineQuantization{}, "");
    accum_scratch_tensor->set_observable(false);
    accum_scratch_tensor->set_data_buffer(nullptr);
    Tensor *accum_scratch =
      helper.getRuntimeGraph(node->graph())->addTensor(std::move(accum_scratch_tensor));
    outputs.push_back(accum_scratch);

    // input zp tensor
    auto input_zp_tensor =
      std::make_unique<Tensor>(DataType::FLOAT32, Shape({}), AffineQuantization{}, "");
    input_zp_tensor->set_observable(false);
    input_zp_tensor->set_data_buffer(nullptr);
    Tensor *input_zp = helper.getRuntimeGraph(node->graph())->addTensor(std::move(input_zp_tensor));
    outputs.push_back(input_zp);

    // output state zp tensor
    auto output_state_zp_tensor =
      std::make_unique<Tensor>(DataType::FLOAT32, Shape({}), AffineQuantization{}, "");
    output_state_zp_tensor->set_observable(false);
    output_state_zp_tensor->set_data_buffer(nullptr);
    Tensor *output_state_zp =
      helper.getRuntimeGraph(node->graph())->addTensor(std::move(output_state_zp_tensor));
    outputs.push_back(output_state_zp);

    // row sums tensor
    auto row_sums_tensor =
      std::make_unique<Tensor>(DataType::S32, Shape({}), AffineQuantization{}, "");
    row_sums_tensor->set_observable(false);
    row_sums_tensor->set_data_buffer(nullptr);
    Tensor *row_sums = helper.getRuntimeGraph(node->graph())->addTensor(std::move(row_sums_tensor));
    outputs.push_back(row_sums);
  }

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
    cell_gate_bias, output_gate_bias, projection_weights, projection_bias, activation_state,
    cell_state, input_layer_norm_coefficients, forget_layer_norm_coefficients,
    cell_layer_norm_coefficients, output_layer_norm_coefficients, std::move(outputs), params);
}

} // namespace luci_interpreter
