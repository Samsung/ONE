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
  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  LUCI_INTERPRETER_CHECK(params().cell_clip >= 0);
  LUCI_INTERPRETER_CHECK(params().proj_clip >= 0);

  if (input_to_input_weights() != nullptr)
  {
    const Shape &input_to_input_weights_shape = input_to_input_weights()->shape();
    LUCI_INTERPRETER_CHECK(input_to_input_weights_shape.num_dims() == 2);
    LUCI_INTERPRETER_CHECK(input_to_input_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(input_to_input_weights_shape.dim(1) == n_input);
  }

  const Shape &input_to_forget_weights_shape = input_to_forget_weights()->shape();
  LUCI_INTERPRETER_CHECK(input_to_forget_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_to_forget_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(input_to_forget_weights_shape.dim(1) == n_input);

  const Shape &input_to_cell_weights_shape = input_to_cell_weights()->shape();
  LUCI_INTERPRETER_CHECK(input_to_cell_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_to_cell_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(input_to_cell_weights_shape.dim(1) == n_input);

  if (recurrent_to_input_weights() != nullptr)
  {
    const Shape &recurrent_to_input_weights_shape = recurrent_to_input_weights()->shape();
    LUCI_INTERPRETER_CHECK(recurrent_to_input_weights_shape.num_dims() == 2);
    LUCI_INTERPRETER_CHECK(recurrent_to_input_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(recurrent_to_input_weights_shape.dim(1) == n_output);
  }

  const Shape &recurrent_to_forget_weights_shape = recurrent_to_forget_weights()->shape();
  LUCI_INTERPRETER_CHECK(recurrent_to_forget_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(recurrent_to_forget_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(recurrent_to_forget_weights_shape.dim(1) == n_output);

  const Shape &recurrent_to_cell_weights_shape = recurrent_to_cell_weights()->shape();
  LUCI_INTERPRETER_CHECK(recurrent_to_cell_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(recurrent_to_cell_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(recurrent_to_cell_weights_shape.dim(1) == n_output);

  // We make sure the input-gate's parameters are either both present (regular
  // LSTM) or not at all (CIFG-LSTM).
  const bool cifg_weights_all_or_none =
    ((input_to_input_weights() != nullptr) && (recurrent_to_input_weights() != nullptr)) ||
    ((input_to_input_weights() == nullptr) && (recurrent_to_input_weights() == nullptr));
  LUCI_INTERPRETER_CHECK(cifg_weights_all_or_none == true);

  if (cell_to_input_weights() != nullptr)
  {
    const Shape &cell_to_input_weights_shape = cell_to_input_weights()->shape();
    LUCI_INTERPRETER_CHECK(cell_to_input_weights_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_to_input_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(is_integer
                             ? cell_to_input_weights()->element_type() == loco::DataType::S16
                             : cell_to_input_weights()->element_type() ==
                                 input_to_forget_weights()->element_type());
  }

  if (cell_to_forget_weights() != nullptr)
  {
    const Shape &cell_to_forget_weights_shape = cell_to_forget_weights()->shape();
    LUCI_INTERPRETER_CHECK(cell_to_forget_weights_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_to_forget_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(is_integer
                             ? cell_to_forget_weights()->element_type() == loco::DataType::S16
                             : cell_to_forget_weights()->element_type() ==
                                 input_to_forget_weights()->element_type());
  }

  if (cell_to_output_weights() != nullptr)
  {
    const Shape &cell_to_output_weights_shape = cell_to_output_weights()->shape();
    LUCI_INTERPRETER_CHECK(cell_to_output_weights_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_to_output_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(is_integer
                             ? cell_to_output_weights()->element_type() == loco::DataType::S16
                             : cell_to_output_weights()->element_type() ==
                                 input_to_forget_weights()->element_type());
  }

  // Making sure the peephole weights are there all or none.
  const bool use_cifg = (input_to_input_weights() == nullptr);
  const bool peephole_weights_all_or_none =
    ((cell_to_input_weights() != nullptr || use_cifg) && (cell_to_forget_weights() != nullptr) &&
     (cell_to_output_weights() != nullptr)) ||
    ((cell_to_input_weights() == nullptr) && (cell_to_forget_weights() == nullptr) &&
     (cell_to_output_weights() == nullptr));
  LUCI_INTERPRETER_CHECK(peephole_weights_all_or_none == true);

  // Make sure the input gate bias is present only when not a CIFG-LSTM.
  if (use_cifg)
  {
    LUCI_INTERPRETER_CHECK(input_gate_bias() == nullptr);
  }
  else
  {
    const Shape &input_gate_bias_shape = input_gate_bias()->shape();
    LUCI_INTERPRETER_CHECK(input_gate_bias_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(input_gate_bias_shape.dim(0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(input_gate_bias()->element_type() == loco::DataType::S32);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(input_gate_bias()->element_type() == loco::DataType::FLOAT32);
    }
  }

  const Shape &forget_gate_bias_shape = forget_gate_bias()->shape();
  LUCI_INTERPRETER_CHECK(forget_gate_bias_shape.num_dims() == 1);
  LUCI_INTERPRETER_CHECK(forget_gate_bias_shape.dim(0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(forget_gate_bias()->element_type() == loco::DataType::S32);
  }
  else
  {
    LUCI_INTERPRETER_CHECK(forget_gate_bias()->element_type() == loco::DataType::FLOAT32);
  }

  const Shape &cell_gate_bias_shape = cell_gate_bias()->shape();
  LUCI_INTERPRETER_CHECK(cell_gate_bias_shape.num_dims() == 1);
  LUCI_INTERPRETER_CHECK(cell_gate_bias_shape.dim(0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(cell_gate_bias()->element_type() == loco::DataType::S32);
  }
  else
  {
    LUCI_INTERPRETER_CHECK(cell_gate_bias()->element_type() == loco::DataType::FLOAT32);
  }

  const Shape &output_gate_bias_shape = output_gate_bias()->shape();
  LUCI_INTERPRETER_CHECK(output_gate_bias_shape.num_dims() == 1);
  LUCI_INTERPRETER_CHECK(output_gate_bias_shape.dim(0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(output_gate_bias()->element_type() == loco::DataType::S32);
  }
  else
  {
    LUCI_INTERPRETER_CHECK(output_gate_bias()->element_type() == loco::DataType::FLOAT32);
  }

  if (projection_weights() != nullptr)
  {
    const Shape &projection_weights_shape = projection_weights()->shape();
    LUCI_INTERPRETER_CHECK(projection_weights_shape.num_dims() == 2);
    LUCI_INTERPRETER_CHECK(projection_weights_shape.dim(0) == n_output);
    LUCI_INTERPRETER_CHECK(projection_weights_shape.dim(1) == n_cell);
  }

  if (projection_bias() != nullptr)
  {
    const Shape &projection_bias_shape = projection_bias()->shape();
    LUCI_INTERPRETER_CHECK(projection_bias_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(projection_bias_shape.dim(0) == n_output);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(projection_bias()->element_type() == loco::DataType::S32);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(projection_bias()->element_type() == loco::DataType::FLOAT32);
    }
  }

  // Making sure the projection tensors are consistent:
  // 1) If projection weight is not present, then projection bias should not be
  // present.
  // 2) If projection weight is present, then projection bias is optional.
  // TODO(ghodrat): make sure this is correct.
  const bool projecton_tensors_consistent =
    ((projection_weights() != nullptr) || (projection_bias() == nullptr));
  LUCI_INTERPRETER_CHECK(projecton_tensors_consistent == true);

  if (use_layer_norm)
  {
    if (use_cifg)
    {
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients() == nullptr);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients() != nullptr)

      const Shape &input_layer_norm_coefficients_shape = input_layer_norm_coefficients()->shape();
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients_shape.num_dims() == 1);
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients_shape.dim(0) == n_cell);
      if (is_integer)
      {
        LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients()->element_type() ==
                               loco::DataType::S16);
      }
      else
      {
        LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients()->element_type() ==
                               loco::DataType::FLOAT32);
      }
    }

    const Shape &forget_layer_norm_coefficients_shape = forget_layer_norm_coefficients()->shape();
    LUCI_INTERPRETER_CHECK(forget_layer_norm_coefficients_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(forget_layer_norm_coefficients_shape.dim(0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(forget_layer_norm_coefficients()->element_type() ==
                             loco::DataType::S16);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(forget_layer_norm_coefficients()->element_type() ==
                             loco::DataType::FLOAT32);
    }

    const Shape &cell_layer_norm_coefficients_shape = cell_layer_norm_coefficients()->shape();
    LUCI_INTERPRETER_CHECK(cell_layer_norm_coefficients_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_layer_norm_coefficients_shape.dim(0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(cell_layer_norm_coefficients()->element_type() == loco::DataType::S16);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(cell_layer_norm_coefficients()->element_type() ==
                             loco::DataType::FLOAT32);
    }

    const Shape &output_layer_norm_coefficients_shape = output_layer_norm_coefficients()->shape();
    LUCI_INTERPRETER_CHECK(output_layer_norm_coefficients_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(output_layer_norm_coefficients_shape.dim(0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(output_layer_norm_coefficients()->element_type() ==
                             loco::DataType::S16);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(output_layer_norm_coefficients()->element_type() ==
                             loco::DataType::FLOAT32);
    }
  }
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
