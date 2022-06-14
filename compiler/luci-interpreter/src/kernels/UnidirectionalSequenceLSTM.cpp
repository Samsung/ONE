/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#include "EvalLSTMHelper.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{
TfLiteFusedActivation get_tflite_activation(Activation activation)
{
  switch (activation)
  {
    case luci::FusedActFunc::RELU:
      return kTfLiteActRelu;
    case luci::FusedActFunc::RELU6:
      return kTfLiteActRelu6;
    case luci::FusedActFunc::RELU_N1_TO_1:
      return kTfLiteActReluN1To1;
    case luci::FusedActFunc::TANH:
      return kTfLiteActTanh;
    case luci::FusedActFunc::SIGN_BIT:
      return kTfLiteActSignBit;
    case luci::FusedActFunc::NONE:
      return kTfLiteActNone;
    default:
      throw std::runtime_error("Unsupported activation type");
  }
}
} // namespace

UnidirectionalSequenceLSTM::UnidirectionalSequenceLSTM(
  const Tensor *input, const Tensor *input_to_input_weights, const Tensor *input_to_forget_weights,
  const Tensor *input_to_cell_weights, const Tensor *input_to_output_weights,
  const Tensor *recurrent_to_input_weights, const Tensor *recurrent_to_forget_weights,
  const Tensor *recurrent_to_cell_weights, const Tensor *recurrent_to_output_weights,
  const Tensor *cell_to_input_weights, const Tensor *cell_to_forget_weights,
  const Tensor *cell_to_output_weights, const Tensor *input_gate_bias,
  const Tensor *forget_gate_bias, const Tensor *cell_gate_bias, const Tensor *output_gate_bias,
  const Tensor *projection_weights, const Tensor *projection_bias, const Tensor *activation_state,
  const Tensor *cell_state, const Tensor *input_layer_norm_coefficients,
  const Tensor *forget_layer_norm_coefficients, const Tensor *cell_layer_norm_coefficients,
  const Tensor *output_layer_norm_coefficients, std::vector<Tensor *> &&outputs,
  const UnidirectionalSequenceLSTMParams &params)
  : KernelWithParams<UnidirectionalSequenceLSTMParams>({input,
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
                                                        activation_state,
                                                        cell_state,
                                                        input_layer_norm_coefficients,
                                                        forget_layer_norm_coefficients,
                                                        cell_layer_norm_coefficients,
                                                        output_layer_norm_coefficients},
                                                       outputs, params)
{
  // Do nothing
}

void UnidirectionalSequenceLSTM::checkInputTensorDimensions(int n_input, int n_output, int n_cell,
                                                            bool is_integer) const
{
  // Making sure clipping parameters have valid values.
  // == 0 means no clipping
  //  > 0 means clipping
  LUCI_INTERPRETER_CHECK(params().cell_clip >= 0);
  LUCI_INTERPRETER_CHECK(params().proj_clip >= 0);

  if (input_to_input_weights() != nullptr)
  {
    const auto input_to_input_weights_shape = input_to_input_weights()->shape();
    LUCI_INTERPRETER_CHECK(input_to_input_weights_shape.num_dims() == 2);
    LUCI_INTERPRETER_CHECK(input_to_input_weights_shape.dim(0) == n_cell);
    LUCI_INTERPRETER_CHECK(input_to_input_weights_shape.dim(1) == n_input);
  }

  const auto input_to_forget_weights_shape = input_to_forget_weights()->shape();
  LUCI_INTERPRETER_CHECK(input_to_forget_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_to_forget_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(input_to_forget_weights_shape.dim(1) == n_input);

  const auto input_to_cell_weights_shape = input_to_cell_weights()->shape();
  LUCI_INTERPRETER_CHECK(input_to_cell_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_to_cell_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(input_to_cell_weights_shape.dim(1) == n_input);

  const auto recurrent_to_input_weights_shape = recurrent_to_input_weights()->shape();
  LUCI_INTERPRETER_CHECK(recurrent_to_input_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(recurrent_to_input_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(recurrent_to_input_weights_shape.dim(1) == n_output);

  const auto recurrent_to_forget_weights_shape = recurrent_to_forget_weights()->shape();
  LUCI_INTERPRETER_CHECK(recurrent_to_forget_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(recurrent_to_forget_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(recurrent_to_forget_weights_shape.dim(1) == n_output);

  const auto recurrent_to_cell_weights_shape = recurrent_to_cell_weights()->shape();
  LUCI_INTERPRETER_CHECK(recurrent_to_cell_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(recurrent_to_cell_weights_shape.dim(0) == n_cell);
  LUCI_INTERPRETER_CHECK(recurrent_to_cell_weights_shape.dim(1) == n_output);

  // We make sure the input-gate's parameters are either both present (regular
  // LSTM) or not at all (CIFG-LSTM).
  const auto cifg_weights_all_or_none =
    ((input_to_input_weights() != nullptr) && (recurrent_to_input_weights() != nullptr)) ||
    ((input_to_input_weights() == nullptr) && (recurrent_to_input_weights() == nullptr));
  LUCI_INTERPRETER_CHECK(cifg_weights_all_or_none == true);

  if (cell_to_input_weights() != nullptr)
  {
    const auto cell_to_input_weights_shape = cell_to_input_weights()->shape();
    LUCI_INTERPRETER_CHECK(cell_to_input_weights_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_to_input_weights_shape.dim(0) == n_cell);
    const auto cell_input_weights_type =
      is_integer ? DataType::S16 : input_to_forget_weights()->element_type();
    LUCI_INTERPRETER_CHECK(cell_to_input_weights()->element_type() == cell_input_weights_type);
  }

  if (cell_to_forget_weights() != nullptr)
  {
    const auto cell_to_forget_weights_shape = cell_to_forget_weights()->shape();
    LUCI_INTERPRETER_CHECK(cell_to_forget_weights_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_to_forget_weights_shape.dim(0) == n_cell);
    const auto cell_forget_weights_type =
      is_integer ? DataType::S16 : input_to_forget_weights()->element_type();
    LUCI_INTERPRETER_CHECK(cell_to_forget_weights()->element_type() == cell_forget_weights_type);
  }

  if (cell_to_output_weights() != nullptr)
  {
    const auto cell_to_output_weights_shape = cell_to_output_weights()->shape();
    LUCI_INTERPRETER_CHECK(cell_to_output_weights_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(cell_to_output_weights_shape.dim(0) == n_cell);
    const auto cell_output_weights_type =
      is_integer ? DataType::S16 : input_to_forget_weights()->element_type();
    LUCI_INTERPRETER_CHECK(cell_to_output_weights()->element_type() == cell_output_weights_type);
  }

  // Making sure the peephole weights are there all or none.
  const auto use_cifg = (input_to_input_weights() == nullptr);
  const auto peephole_weights_all_or_none =
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
    LUCI_INTERPRETER_CHECK(input_gate_bias() != nullptr);

    const auto input_gate_bias_shape = input_gate_bias()->shape();
    LUCI_INTERPRETER_CHECK(input_gate_bias_shape.num_dims() == 1);
    LUCI_INTERPRETER_CHECK(input_gate_bias_shape.dim(0) == n_cell);
    if (is_integer)
    {
      LUCI_INTERPRETER_CHECK(input_gate_bias()->element_type() == loco::DataType::S32)
    }
    else
    {
      LUCI_INTERPRETER_CHECK(input_gate_bias()->element_type() == loco::DataType::FLOAT32)
    }
  }

  const auto forget_gate_bias_shape = forget_gate_bias()->shape();
  LUCI_INTERPRETER_CHECK(forget_gate_bias_shape.num_dims() == 1);
  LUCI_INTERPRETER_CHECK(forget_gate_bias_shape.dim(0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(forget_gate_bias()->element_type() == loco::DataType::S32)
  }
  else
  {
    LUCI_INTERPRETER_CHECK(forget_gate_bias()->element_type() == loco::DataType::FLOAT32)
  }

  const auto cell_gate_bias_shape = cell_gate_bias()->shape();
  LUCI_INTERPRETER_CHECK(cell_gate_bias_shape.num_dims() == 1);
  LUCI_INTERPRETER_CHECK(cell_gate_bias_shape.dim(0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(cell_gate_bias()->element_type() == loco::DataType::S32)
  }
  else
  {
    LUCI_INTERPRETER_CHECK(cell_gate_bias()->element_type() == loco::DataType::FLOAT32)
  }

  const auto output_gate_bias_shape = output_gate_bias()->shape();
  LUCI_INTERPRETER_CHECK(output_gate_bias_shape.num_dims() == 1);
  LUCI_INTERPRETER_CHECK(output_gate_bias_shape.dim(0) == n_cell);
  if (is_integer)
  {
    LUCI_INTERPRETER_CHECK(output_gate_bias()->element_type() == loco::DataType::S32)
  }
  else
  {
    LUCI_INTERPRETER_CHECK(output_gate_bias()->element_type() == loco::DataType::FLOAT32)
  }

  if (projection_weights() != nullptr)
  {
    const auto projection_weights_shape = projection_weights()->shape();
    LUCI_INTERPRETER_CHECK(projection_weights_shape.num_dims() == 2);
    LUCI_INTERPRETER_CHECK(projection_weights_shape.dim(0) == n_output);
    LUCI_INTERPRETER_CHECK(projection_weights_shape.dim(1) == n_cell);
  }

  if (projection_bias() != nullptr)
  {
    const auto projection_bias_shape = projection_bias()->shape();
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
  const auto projecton_tensors_consistent =
    ((projection_weights() != nullptr) || (projection_bias() == nullptr));
  LUCI_INTERPRETER_CHECK(projecton_tensors_consistent == true);

  if (forget_layer_norm_coefficients() != nullptr)
  {
    if (use_cifg)
    {
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients() == nullptr);
    }
    else
    {
      LUCI_INTERPRETER_CHECK(input_layer_norm_coefficients() != nullptr);
      const auto input_layer_norm_coefficients_shape = input_layer_norm_coefficients()->shape();
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

    const auto forget_layer_norm_coefficients_shape = forget_layer_norm_coefficients()->shape();
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

    const auto cell_layer_norm_coefficients_shape = cell_layer_norm_coefficients()->shape();
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

    const auto output_layer_norm_coefficients_shape = output_layer_norm_coefficients()->shape();
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

// Resize the output and  state tensors based on the sizes of the input tensors.
// Allocate a temporary scratch tensor. Also check that the sizes of the input
// tensors match each other.
void UnidirectionalSequenceLSTM::configure()
{
  auto use_layer_norm = false;
  if (forget_layer_norm_coefficients() != nullptr)
    use_layer_norm = true;

  // Inferring batch size, number of outputs and sequence length and
  // number of cells from the input tensors.
  const auto is_integer = input()->element_type() == loco::DataType::S8;
  if (is_integer)
  {
    // TODO: Support full integer case(need to support intermediate tensors)
    throw std::runtime_error("Not supported yet");
  }

  const auto input_shape = input()->shape();
  LUCI_INTERPRETER_CHECK(input_shape.num_dims() > 1);
  const auto time_major = params().time_major;
  const auto n_batch = time_major ? input_shape.dim(1) : input_shape.dim(0);
  const auto n_input = input_shape.dim(2);

  const auto input_to_output_weights_shape = input_to_output_weights()->shape();
  LUCI_INTERPRETER_CHECK(input_to_output_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(input_to_output_weights_shape.dim(1) == n_input);
  const auto n_cell = input_to_output_weights_shape.dim(0);

  const auto recurrent_to_output_weights_shape = recurrent_to_output_weights()->shape();
  LUCI_INTERPRETER_CHECK(recurrent_to_output_weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(recurrent_to_output_weights_shape.dim(0) == n_cell);
  const auto n_output = recurrent_to_output_weights_shape.dim(1);

  // Check that input tensor dimensions matches with each other.
  checkInputTensorDimensions(n_input, n_output, n_cell, is_integer);

  // Check the pointer to output_state and cell_state tensors.
  LUCI_INTERPRETER_CHECK(output_state() != nullptr);
  LUCI_INTERPRETER_CHECK(cell_state() != nullptr);

  // Check the shape of input state tensors.
  // These tensor may be 1D or 2D. It's fine as long as the total size is
  // correct.
  LUCI_INTERPRETER_CHECK(output_state()->shape().num_elements() == n_batch * n_output);
  LUCI_INTERPRETER_CHECK(cell_state()->shape().num_elements() == n_batch * n_cell);

  // Resize the output tensors.
  Shape output_shape = input_shape;
  output_shape.dim(input_shape.num_dims() - 1) = n_output;
  output()->resize(output_shape);

  // Resize scratch_buffer
  const auto use_cifg = (input_to_input_weights() == nullptr);
  Shape scratch_buffer_shape(2);
  scratch_buffer_shape.dim(0) = n_batch;
  if (use_cifg)
  {
    // Reserving space for Cell, Forget, Output gates
    scratch_buffer_shape.dim(1) = n_cell * 3;
  }
  else
  {
    // Reserving space for Input, Cell, Forget, Output gates
    scratch_buffer_shape.dim(1) = n_cell * 4;
  }
  scratch_buffer()->resize(scratch_buffer_shape);

  // If is hybrid
  if ((input_to_output_weights()->element_type() == loco::DataType::U8 ||
       input_to_output_weights()->element_type() == loco::DataType::S8) &&
      input()->element_type() == loco::DataType::FLOAT32)
  {
    _compute_row_sums = true;
    // Allocate temporary tensors to store quantized values of input,
    // output_state and cell_state tensors.
    input_quantized()->resize(input_shape);

    output_state_quantized()->resize(output_state()->shape());

    cell_state_quantized()->resize(cell_state()->shape());

    input_sf()->resize({n_batch});

    output_state_sf()->resize({n_batch});

    prod_scaling_factors()->resize({n_batch});

    // Allocate a temporary tensor to store the recovered cell weights. Since
    // this is used for diagonal matrices, only need to store n_cell values.
    recovered_cell_weights()->resize({n_cell});

    // Allocate a temporary tensor to store the accumulated int32 values.
    Shape accum_scratch_shape(2);
    accum_scratch_shape.dim(0) = n_cell;
    accum_scratch_shape.dim(1) = n_batch;
    accum_scratch()->resize(accum_scratch_shape);

    input_zp()->resize({n_batch});

    output_state_zp()->resize({n_batch});

    auto row_sums_rows = use_cifg ? 6 : 8;
    if (projection_weights() != nullptr)
    {
      row_sums_rows += ceil(static_cast<float>(n_output) / n_cell);
    }
    Shape row_sums_shape(2);
    row_sums_shape.dim(0) = row_sums_rows;
    row_sums_shape.dim(1) = n_cell;
    row_sums()->resize(row_sums_shape);
  }
}

void UnidirectionalSequenceLSTM::execute() const
{
  switch (input_to_output_weights()->element_type())
  {
    case loco::DataType::FLOAT32:
    {
      evalFloat();
      break;
    }
    case loco::DataType::S8:
    case loco::DataType::U8:
    {
      // If is hybrid
      if (input()->element_type() == loco::DataType::FLOAT32)
      {
        evalHybrid();
        break;
      }
    }
    default:
      throw std::runtime_error("unsupported yet");
  }
}

void UnidirectionalSequenceLSTM::evalHybrid() const
{
  const auto input_shape = input()->shape();
  const int n_input = input_shape.dim(input_shape.num_dims() - 1);
  int32_t max_time, n_batch;

  auto compute_row_sums = _compute_row_sums;
  if (input_shape.num_dims() == 2)
  {
    max_time = 1;
    n_batch = input_shape.dim(0);
  }
  else
  {
    max_time = params().time_major ? input_shape.dim(0) : input_shape.dim(1);
    n_batch = params().time_major ? input_shape.dim(1) : input_shape.dim(0);
  }
  const auto aux_input_size = 0;
  // n_cell and n_output will be the same size when there is no projection.
  const auto n_cell = input_to_output_weights()->shape().dim(0);
  const auto n_output = recurrent_to_output_weights()->shape().dim(1);

  const auto row_sums_size = row_sums()->shape().dim(0);

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to get the condition.
  const auto use_cifg = (input_to_input_weights() == nullptr);

  auto scratch_buffer_ptr = getTensorData<float>(scratch_buffer());
  float *input_gate_scratch = nullptr;
  float *cell_gate_scratch = nullptr;
  float *forget_gate_scratch = nullptr;
  float *output_gate_scratch = nullptr;
  if (use_cifg)
  {
    cell_gate_scratch = scratch_buffer_ptr;
    forget_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
  }
  else
  {
    input_gate_scratch = scratch_buffer_ptr;
    cell_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 3 * n_cell * n_batch;
  }

  const auto output_batch_leading_dim = output()->shape().dim(output()->shape().num_dims() - 1);
  int32_t *input_zp_ptr = nullptr;
  int32_t *aux_input_zp_ptr = nullptr;
  int32_t *output_state_zp_ptr = nullptr;
  int32_t *row_sums_ptr = nullptr;
  if (params().asymmetric_quantize_inputs)
  {
    input_zp_ptr = getTensorData<int32_t>(input_zp());
    aux_input_zp_ptr = nullptr;
    output_state_zp_ptr = getTensorData<int32_t>(output_state_zp());
    row_sums_ptr = getTensorData<int32_t>(row_sums());
  }

  TfLiteLSTMParams lstm_params;
  lstm_params.activation = get_tflite_activation(params().activation);
  lstm_params.cell_clip = params().cell_clip;
  lstm_params.proj_clip = params().proj_clip;
  lstm_params.asymmetric_quantize_inputs = params().asymmetric_quantize_inputs;

  if (params().time_major)
  {
    // Feed the sequence into the LSTM step-by-step.
    const auto input_step = n_batch * n_input;
    const auto output_step = n_batch * output_batch_leading_dim;
    for (int32_t t = 0; t < max_time; t++)
    {
      const auto input_ptr = getTensorData<float>(input()) + t * input_step;
      const auto output_ptr = getTensorData<float>(output()) + t * output_step;
      eval_lstm::LstmStepHybrid(
        input_ptr, getTensorData<int8_t>(input_to_input_weights()),
        /*input_to_input_weights_ledger*/ nullptr, getTensorScale(input_to_input_weights()),
        getTensorData<int8_t>(input_to_forget_weights()),
        /*input_to_forget_weights_ledger*/ nullptr, getTensorScale(input_to_forget_weights()),
        getTensorData<int8_t>(input_to_cell_weights()),
        /*input_to_cell_weights_ledger*/ nullptr, getTensorScale(input_to_cell_weights()),
        getTensorData<int8_t>(input_to_output_weights()),
        /*input_to_output_weights_ledger*/ nullptr, getTensorScale(input_to_output_weights()),
        /*aux_input_ptr*/ nullptr,
        /*aux_input_to_input_weights*/ nullptr,
        /*scale for aux_input_to_input_weights*/ 1.0f,
        /*aux_input_to_forget_weights*/ nullptr,
        /*scale for aux_input_to_forget_weights*/ 1.0f,
        /*aux_input_to_cell_weights*/ nullptr,
        /*scale for aux_input_to_cell_weights*/ 1.0f,
        /*aux_input_to_output_weights*/ nullptr,
        /*scale for aux_input_to_output_weights*/ 1.0f,
        getTensorData<int8_t>(recurrent_to_input_weights()),
        /*recurrent_to_input_weights_ledger*/ nullptr, getTensorScale(recurrent_to_input_weights()),
        getTensorData<int8_t>(recurrent_to_forget_weights()),
        /*recurrent_to_forget_weights_ledger*/ nullptr,
        getTensorScale(recurrent_to_forget_weights()),
        getTensorData<int8_t>(recurrent_to_cell_weights()),
        /*recurrent_to_cell_weights_ledger*/ nullptr, getTensorScale(recurrent_to_cell_weights()),
        getTensorData<int8_t>(recurrent_to_output_weights()),
        /*recurrent_to_output_weights_ledger*/ nullptr,
        getTensorScale(recurrent_to_output_weights()),
        getTensorData<int8_t>(cell_to_input_weights()), getTensorScale(cell_to_input_weights()),
        getTensorData<int8_t>(cell_to_forget_weights()), getTensorScale(cell_to_forget_weights()),
        getTensorData<int8_t>(cell_to_output_weights()), getTensorScale(cell_to_output_weights()),
        getTensorData<float>(input_layer_norm_coefficients()),
        getTensorData<float>(forget_layer_norm_coefficients()),
        getTensorData<float>(cell_layer_norm_coefficients()),
        getTensorData<float>(output_layer_norm_coefficients()),
        getTensorData<float>(input_gate_bias()), getTensorData<float>(forget_gate_bias()),
        getTensorData<float>(cell_gate_bias()), getTensorData<float>(output_gate_bias()),
        getTensorData<int8_t>(projection_weights()),
        /*projection_weights_ledger*/ nullptr, getTensorScale(projection_weights()),
        getTensorData<float>(projection_bias()), const_cast<const TfLiteLSTMParams *>(&lstm_params),
        n_batch, n_cell, n_input, aux_input_size, n_output, output_batch_leading_dim,
        input_gate_scratch, forget_gate_scratch, cell_gate_scratch, output_gate_scratch,
        getTensorData<float>(input_sf()),
        /*aux_input_sf*/ nullptr, getTensorData<float>(output_state_sf()),
        getTensorData<float>(prod_scaling_factors()),
        getTensorData<float>(recovered_cell_weights()), getTensorData<int8_t>(input_quantized()),
        /*aux_input_quantized*/ nullptr, getTensorData<int8_t>(output_state_quantized()),
        getTensorData<int8_t>(cell_state_quantized()),
        const_cast<float *>(getTensorData<float>(output_state())),
        const_cast<float *>(getTensorData<float>(cell_state())),
        getTensorData<int32_t>(accum_scratch()), output_ptr, input_zp_ptr, aux_input_zp_ptr,
        output_state_zp_ptr, row_sums_ptr, row_sums_size, &compute_row_sums,
        params().asymmetric_quantize_inputs);
    }
  }
  else
  {
    for (int32_t b = 0; b < n_batch; b++)
    {
      const auto input_step = n_input;
      const auto output_step = output_batch_leading_dim;
      for (int32_t t = 0; t < max_time; t++)
      {
        const auto time_offset = b * max_time + t;
        const auto input_ptr = getTensorData<float>(input()) + time_offset * input_step;

        auto output_ptr = getTensorData<float>(output()) + time_offset * output_step;

        // Offset the {output,cell}_state pointers to the right batch.
        auto output_state_ptr =
          const_cast<float *>(getTensorData<float>(output_state())) + b * output_batch_leading_dim;
        auto cell_state_ptr = const_cast<float *>(getTensorData<float>(cell_state())) + b * n_cell;

        // Offset the scratch pointers to the right batch.
        auto input_gate_scratch_ptr =
          input_gate_scratch ? input_gate_scratch + b * n_cell : nullptr;
        auto forget_gate_scratch_ptr = forget_gate_scratch + b * n_cell;
        auto cell_gate_scratch_ptr = cell_gate_scratch + b * n_cell;
        auto output_gate_scratch_ptr = output_gate_scratch + b * n_cell;

        eval_lstm::LstmStepHybrid(
          input_ptr, getTensorData<int8_t>(input_to_input_weights()),
          /*input_to_input_weights_ledger*/ nullptr, getTensorScale(input_to_input_weights()),
          getTensorData<int8_t>(input_to_forget_weights()),
          /*input_to_forget_weights_ledger*/ nullptr, getTensorScale(input_to_forget_weights()),
          getTensorData<int8_t>(input_to_cell_weights()),
          /*input_to_cell_weights_ledger*/ nullptr, getTensorScale(input_to_cell_weights()),
          getTensorData<int8_t>(input_to_output_weights()),
          /*input_to_output_weights_ledger*/ nullptr, getTensorScale(input_to_output_weights()),
          /*aux_input_ptr*/ nullptr,
          /*aux_input_to_input_weights*/ nullptr,
          /*scale for aux_input_to_input_weights*/ 1.0f,
          /*aux_input_to_forget_weights*/ nullptr,
          /*scale for aux_input_to_forget_weights*/ 1.0f,
          /*aux_input_to_cell_weights*/ nullptr,
          /*scale for aux_input_to_cell_weights*/ 1.0f,
          /*aux_input_to_output_weights*/ nullptr,
          /*scale for aux_input_to_output_weights*/ 1.0f,
          getTensorData<int8_t>(recurrent_to_input_weights()),
          /*recurrent_to_input_weights_ledger*/ nullptr,
          getTensorScale(recurrent_to_input_weights()),
          getTensorData<int8_t>(recurrent_to_forget_weights()),
          /*recurrent_to_forget_weights_ledger*/ nullptr,
          getTensorScale(recurrent_to_forget_weights()),
          getTensorData<int8_t>(recurrent_to_cell_weights()),
          /*recurrent_to_cell_weights_ledger*/ nullptr, getTensorScale(recurrent_to_cell_weights()),
          getTensorData<int8_t>(recurrent_to_output_weights()),
          /*recurrent_to_output_weights_ledger*/ nullptr,
          getTensorScale(recurrent_to_output_weights()),
          getTensorData<int8_t>(cell_to_input_weights()), getTensorScale(cell_to_input_weights()),
          getTensorData<int8_t>(cell_to_forget_weights()), getTensorScale(cell_to_forget_weights()),
          getTensorData<int8_t>(cell_to_output_weights()), getTensorScale(cell_to_output_weights()),
          getTensorData<float>(input_layer_norm_coefficients()),
          getTensorData<float>(forget_layer_norm_coefficients()),
          getTensorData<float>(cell_layer_norm_coefficients()),
          getTensorData<float>(output_layer_norm_coefficients()),
          getTensorData<float>(input_gate_bias()), getTensorData<float>(forget_gate_bias()),
          getTensorData<float>(cell_gate_bias()), getTensorData<float>(output_gate_bias()),
          getTensorData<int8_t>(projection_weights()),
          /*projection_weights_ledger*/ nullptr, getTensorScale(projection_weights()),
          getTensorData<float>(projection_bias()),
          const_cast<const TfLiteLSTMParams *>(&lstm_params), /*n_batch=*/1, n_cell, n_input,
          aux_input_size, n_output, output_batch_leading_dim, input_gate_scratch_ptr,
          forget_gate_scratch, cell_gate_scratch_ptr, output_gate_scratch_ptr,
          getTensorData<float>(input_sf()),
          /*aux_input_sf*/ nullptr, getTensorData<float>(output_state_sf()),
          getTensorData<float>(prod_scaling_factors()),
          getTensorData<float>(recovered_cell_weights()), getTensorData<int8_t>(input_quantized()),
          /*aux_input_quantized*/ nullptr, getTensorData<int8_t>(output_state_quantized()),
          getTensorData<int8_t>(cell_state_quantized()), output_state_ptr, cell_state_ptr,
          getTensorData<int32_t>(accum_scratch()), output_ptr, input_zp_ptr, aux_input_zp_ptr,
          output_state_zp_ptr, row_sums_ptr, row_sums_size, &compute_row_sums,
          params().asymmetric_quantize_inputs);
      }
    }
  }
}

void UnidirectionalSequenceLSTM::evalFloat() const
{
  int max_time, n_batch;
  if (input()->shape().num_dims() == 3)
  {
    max_time = (params().time_major) ? input()->shape().dim(0) : input()->shape().dim(1);
    n_batch = (params().time_major) ? input()->shape().dim(1) : input()->shape().dim(0);
  }
  else
  {
    max_time = 1;
    n_batch = input()->shape().dim(0);
  }
  const auto n_input = input()->shape().dim(input()->shape().num_dims() - 1);
  const auto aux_input_size = 0;

  // n_cell and n_output will be the same size when there is no projection.
  const auto n_cell = input_to_output_weights()->shape().dim(0);
  const auto n_output = recurrent_to_output_weights()->shape().dim(1);

  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to the get the condition.
  const auto use_cifg = (input_to_input_weights() == nullptr);

  auto scratch_buffer_ptr = getTensorData<float>(scratch_buffer());
  float *input_gate_scratch = nullptr;
  float *cell_gate_scratch = nullptr;
  float *forget_gate_scratch = nullptr;
  float *output_gate_scratch = nullptr;
  if (use_cifg)
  {
    cell_gate_scratch = scratch_buffer_ptr;
    forget_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
  }
  else
  {
    input_gate_scratch = scratch_buffer_ptr;
    cell_gate_scratch = scratch_buffer_ptr + n_cell * n_batch;
    forget_gate_scratch = scratch_buffer_ptr + 2 * n_cell * n_batch;
    output_gate_scratch = scratch_buffer_ptr + 3 * n_cell * n_batch;
  }

  TfLiteLSTMParams lstm_params;
  lstm_params.activation = get_tflite_activation(params().activation);
  lstm_params.cell_clip = params().cell_clip;
  lstm_params.proj_clip = params().proj_clip;
  lstm_params.asymmetric_quantize_inputs = params().asymmetric_quantize_inputs;

  const auto output_batch_leading_dim = output()->shape().dim(output()->shape().num_dims() - 1);
  if (params().time_major)
  {
    // Loop through the sequence.
    const auto input_step = n_batch * n_input;
    const auto output_step = n_batch * output_batch_leading_dim;
    for (int32_t t = 0; t < max_time; t++)
    {
      const auto input_ptr = getTensorData<float>(input()) + t * input_step;
      auto output_ptr = getTensorData<float>(output()) + t * output_step;
      eval_lstm::LstmStepFloat(
        input_ptr, getTensorData<float>(input_to_input_weights()),
        getTensorData<float>(input_to_forget_weights()),
        getTensorData<float>(input_to_cell_weights()),
        getTensorData<float>(input_to_output_weights()),
        /*aux_input_ptr*/ nullptr,
        /*aux_input_to_input_weights*/ nullptr,
        /*aux_input_to_forget_weights*/ nullptr,
        /*aux_input_to_cell_weights*/ nullptr,
        /*aux_input_to_output_weights*/ nullptr, getTensorData<float>(recurrent_to_input_weights()),
        getTensorData<float>(recurrent_to_forget_weights()),
        getTensorData<float>(recurrent_to_cell_weights()),
        getTensorData<float>(recurrent_to_output_weights()),
        getTensorData<float>(cell_to_input_weights()),
        getTensorData<float>(cell_to_forget_weights()),
        getTensorData<float>(cell_to_output_weights()),
        getTensorData<float>(input_layer_norm_coefficients()),
        getTensorData<float>(forget_layer_norm_coefficients()),
        getTensorData<float>(cell_layer_norm_coefficients()),
        getTensorData<float>(output_layer_norm_coefficients()),
        getTensorData<float>(input_gate_bias()), getTensorData<float>(forget_gate_bias()),
        getTensorData<float>(cell_gate_bias()), getTensorData<float>(output_gate_bias()),
        getTensorData<float>(projection_weights()), getTensorData<float>(projection_bias()),
        const_cast<const TfLiteLSTMParams *>(&lstm_params), n_batch, n_cell, n_input,
        aux_input_size, n_output, output_batch_leading_dim,
        const_cast<float *>(getTensorData<float>(output_state())),
        const_cast<float *>(getTensorData<float>(cell_state())), input_gate_scratch,
        forget_gate_scratch, cell_gate_scratch, output_gate_scratch, output_ptr);
    }
  }
  else
  {
    for (int32_t b = 0; b < n_batch; b++)
    {
      const auto input_step = n_input;
      const auto output_step = output_batch_leading_dim;
      for (int32_t t = 0; t < max_time; t++)
      {
        const auto time_offset = b * max_time + t;
        const auto input_ptr = getTensorData<float>(input()) + time_offset * input_step;
        auto output_ptr = getTensorData<float>(output()) + time_offset * output_step;

        auto output_state_ptr =
          const_cast<float *>(getTensorData<float>(output_state()) + b * output_batch_leading_dim);
        auto cell_state_ptr = const_cast<float *>(getTensorData<float>(cell_state()) + b * n_cell);

        auto input_gate_scratch_ptr =
          input_gate_scratch ? input_gate_scratch + b * n_cell : nullptr;
        auto forget_gate_scratch_ptr = forget_gate_scratch + b * n_cell;
        auto cell_gate_scratch_ptr = cell_gate_scratch + b * n_cell;
        auto output_gate_scratch_ptr = output_gate_scratch + b * n_cell;

        eval_lstm::LstmStepFloat(
          input_ptr, getTensorData<float>(input_to_input_weights()),
          getTensorData<float>(input_to_forget_weights()),
          getTensorData<float>(input_to_cell_weights()),
          getTensorData<float>(input_to_output_weights()),
          /*aux_input_ptr*/ nullptr,
          /*aux_input_to_input_weights*/ nullptr,
          /*aux_input_to_forget_weights*/ nullptr,
          /*aux_input_to_cell_weights*/ nullptr,
          /*aux_input_to_output_weights*/ nullptr,
          getTensorData<float>(recurrent_to_input_weights()),
          getTensorData<float>(recurrent_to_forget_weights()),
          getTensorData<float>(recurrent_to_cell_weights()),
          getTensorData<float>(recurrent_to_output_weights()),
          getTensorData<float>(cell_to_input_weights()),
          getTensorData<float>(cell_to_forget_weights()),
          getTensorData<float>(cell_to_output_weights()),
          getTensorData<float>(input_layer_norm_coefficients()),
          getTensorData<float>(forget_layer_norm_coefficients()),
          getTensorData<float>(cell_layer_norm_coefficients()),
          getTensorData<float>(output_layer_norm_coefficients()),
          getTensorData<float>(input_gate_bias()), getTensorData<float>(forget_gate_bias()),
          getTensorData<float>(cell_gate_bias()), getTensorData<float>(output_gate_bias()),
          getTensorData<float>(projection_weights()), getTensorData<float>(projection_bias()),
          const_cast<const TfLiteLSTMParams *>(&lstm_params), /*n_batch=*/n_batch, n_cell, n_input,
          aux_input_size, n_output, output_batch_leading_dim, output_state_ptr, cell_state_ptr,
          input_gate_scratch, forget_gate_scratch, cell_gate_scratch, output_gate_scratch,
          output_ptr);
      }
    }
  }
}

} // namespace kernels
} // namespace luci_interpreter
