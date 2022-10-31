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

#include "kernels/UnidirectionalSequenceLSTM.h"
#include "kernels/Utils.h"
#include "PALUnidirectionalSequenceLSTM.h"

namespace luci_interpreter
{
namespace kernels
{

UnidirectionalSequenceLSTM::UnidirectionalSequenceLSTM(
  const Tensor *input,

  const Tensor *input_to_input_weights, const Tensor *input_to_forget_weights,
  const Tensor *input_to_cell_weights, const Tensor *input_to_output_weights,

  const Tensor *recurrent_to_input_weights, const Tensor *recurrent_to_forget_weights,
  const Tensor *recurrent_to_cell_weights, const Tensor *recurrent_to_output_weights,

  const Tensor *cell_to_input_weights, const Tensor *cell_to_forget_weights,
  const Tensor *cell_to_output_weights,

  const Tensor *input_gate_bias, const Tensor *forget_gate_bias, const Tensor *cell_gate_bias,
  const Tensor *output_gate_bias,

  const Tensor *projection_weights, const Tensor *projection_bias,

  const Tensor *output_state, const Tensor *cell_state, const Tensor *input_layer_norm_coefficients,
  const Tensor *forget_layer_norm_coefficients, const Tensor *cell_layer_norm_coefficients,
  const Tensor *output_layer_norm_coefficients,

  Tensor *output, Tensor *scratchpad_1, Tensor *scratchpad_2, Tensor *scratchpad_3,
  const UnidirectionalSequenceLSTMParams &params)
  : KernelWithParams<UnidirectionalSequenceLSTMParams>(
      {input,
       input_to_input_weights,
       input_to_forget_weights,
       input_to_cell_weights,
       input_to_output_weights,

       recurrent_to_input_weights,
       recurrent_to_forget_weights,
       recurrent_to_cell_weights,
       recurrent_to_output_weights,

       cell_to_input_weights,
       cell_to_forget_weights,
       cell_to_output_weights,

       input_gate_bias,
       forget_gate_bias,
       cell_gate_bias,
       output_gate_bias,

       projection_weights,
       projection_bias,

       output_state,
       cell_state,

       input_layer_norm_coefficients,
       forget_layer_norm_coefficients,
       cell_layer_norm_coefficients,
       output_layer_norm_coefficients},
      {output, scratchpad_1, scratchpad_2, scratchpad_3}, params)
{
  // Do nothing
}

// Check that input tensor dimensions matches with each other.
void UnidirectionalSequenceLSTM::check_input_tensor_dimensions(int n_input, int n_output,
                                                               int n_cell, bool use_layer_norm,
                                                               bool is_integer)
{
  // TODO implement
  (void)n_input;
  (void)n_output;
  (void)n_cell;
  (void)use_layer_norm;
  (void)is_integer;
}

void UnidirectionalSequenceLSTM::configure()
{
  LUCI_INTERPRETER_CHECK(getInputTensors().size() == 24);
  LUCI_INTERPRETER_CHECK(getOutputTensors().size() >= 1);

  // TODO support U8
  LUCI_INTERPRETER_CHECK(input()->element_type() == loco::DataType::FLOAT32);
  const bool is_integer = false;
  const bool use_layer_norm = (forget_layer_norm_coefficients() != nullptr);

  // Inferring batch size, number of outputs and sequence length and
  // number of cells from the input tensors.
  const Shape &input_shape = input()->shape();
  LUCI_INTERPRETER_CHECK(input_shape.num_dims() > 1);
  const bool time_major = params().time_major;
  const int n_batch = time_major ? input_shape.dim(1) : input_shape.dim(0);
  // NOTE as dim(2) is accessed, we need to check this is valid
  LUCI_INTERPRETER_CHECK(input_shape.num_dims() > 2);
  const int n_input = input_shape.dim(2);

  const Shape &input_to_output_weights_shape = input_to_output_weights()->shape();
  const int n_cell = input_to_output_weights_shape.dim(0);
  LUCI_INTERPRETER_CHECK(input_to_output_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_to_output_weights_shape.dim(1) == n_input);

  const Shape &recurrent_to_output_weights_shape = recurrent_to_output_weights()->shape();
  LUCI_INTERPRETER_CHECK(recurrent_to_output_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(recurrent_to_output_weights_shape.dim(0) == n_cell);

  const int n_output = recurrent_to_output_weights_shape.dim(1);

  // Check that input tensor dimensions matches with each other.
  check_input_tensor_dimensions(n_input, n_output, n_cell, use_layer_norm, is_integer);

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  const Shape &output_state_shape = output_state()->shape();
  const Shape &cell_state_shape = cell_state()->shape();
  LUCI_INTERPRETER_CHECK(output_state_shape.num_elements() == n_batch * n_output);
  LUCI_INTERPRETER_CHECK(cell_state_shape.num_elements() == n_batch * n_cell);

  // Resize the output tensors.
  Shape output_shape = Shape(input_shape.num_dims());
  for (int i = 0; i < input_shape.num_dims() - 1; i++)
  {
    output_shape.dim(i) = input_shape.dim(i);
  }
  output_shape.dim(input_shape.num_dims() - 1) = n_output;
  output()->resize(output_shape);

  // TODO import integer

  // output_state and cell_state are variable tensor; use scratchpad.
  getOutputTensors()[1]->resize(output_state_shape);
  getOutputTensors()[2]->resize(cell_state_shape);

  const bool use_cifg = (input_to_input_weights() == nullptr);
  luci_interpreter_pal::SetupScratchpadTensor(getOutputTensors()[3], use_cifg, n_batch, n_cell);

  // hybrid not supported
  if (input_to_output_weights()->element_type() == loco::DataType::U8 &&
      input()->element_type() == loco::DataType::FLOAT32)
  {
    throw std::runtime_error("Hybrid type is not currently supported");
  }
  // TODO support hybrid
  // TODO support U8
}

void UnidirectionalSequenceLSTM::execute() const
{
  switch (input()->element_type())
  {
    case loco::DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type");
  }
}

void UnidirectionalSequenceLSTM::evalFloat() const
{
  // TODO implement
}

} // namespace kernels
} // namespace luci_interpreter
