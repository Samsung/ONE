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

#ifndef LUCI_INTERPRETER_KERNELS_EVAL_LSTM_HELPER_H
#define LUCI_INTERPRETER_KERNELS_EVAL_LSTM_HELPER_H

#include <tensorflow/lite/kernels/internal/tensor_utils.h>

namespace luci_interpreter
{
namespace kernels
{
/*
 *
 * All functions in this file help implement LSTM kernel.
 * They are a copy from tensorflow/lite/kernels/lstm_eval.cc
 *
 */
namespace eval_lstm
{
// Performs an LSTM batch inference step for input specified by input_ptr.
// The LSTM cell is specified by the pointers to its weights (*_weights_ptr) and
// biases (*_bias_ptr), and buffers (*_scratch), along with additional
// parameters:
//  - params: various LSTM params including activation, clipping, etc.,
//  - n_batch: size of batch,
//  - n_cell: number of cells (or units),
//  - n_input: the input size,
//  - n_aux_input: the auxiliary input size.
//  - n_output: the output size.
//  - output_batch_leading_dim: the leading dimension of the output buffer.
//
// Input of size 'n_batch * n_input':
//   input_ptr
// Input of size 'n_batch * n_aux_input':
//   aux_input_ptr                     - optional (can be nullptr)
//
// LSTM weights:
// Input weights of size 'n_cell * n_input':
//   input_to_input_weights            - optional
//   input_to_forget_weights
//   input_to_cell_weights
//   input_to_output_weights
// Auxiliary input weights of size 'n_cell * n_aux_input':
//   aux_input_to_input_weights        - optional
//   aux_input_to_forget_weights       - optional
//   aux_input_to_cell_weights         - optional
//   aux_input_to_output_weights       - optional
// Recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weights        - optional
//   recurrent_to_forget_weights
//   recurrent_to_cell_weights
//   recurrent_to_input_weights
// Peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights             - optional
//   cell_to_cell_weights              - optional
//   cell_to_output_weights            - optional
// Projection weights of size 'n_output * n_cell'
//   projection_weights_ptr            - optional
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr               - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Layer norm coefficients of size 'n_cell', representing diagonal matrices.
//   input_layer_norm_coefficients_ptr  - optional
//   forget_layer_norm_coefficients_ptr - optional
//   cell_layer_norm_coefficients_ptr   - optional
//   output_layer_norm_coefficients_ptr - optional
//
// The pointers to the cell and output state and the output are updated.
//
// The pointers input_ptr, aux_input_ptr, and output_ptr point to data aligned
// in batch_major order, and each step processes batch_size many inputs from
// input_ptr, and updates batch_size many cell and output states.
//
// The output_batch_dim is output.shape[-1], i.e. the outermost dimension of the
// output tensor, and in most cases will be equal to n_output. It is usually not
// when we want to store the LSTM output into a slice of the output tensor, e.g.
// for bidirectional LSTMs with merge_outputs. In this case, the batched
// operations cannot be used since they assume that the batched outputs are
// contiguous, and we manually loop over the batched outputs.
void LstmStepFloat(
  const float *input_ptr, const float *input_to_input_weights_ptr,
  const float *input_to_forget_weights_ptr, const float *input_to_cell_weights_ptr,
  const float *input_to_output_weights_ptr, const float *aux_input_ptr,
  const float *aux_input_to_input_weights_ptr, const float *aux_input_to_forget_weights_ptr,
  const float *aux_input_to_cell_weights_ptr, const float *aux_input_to_output_weights_ptr,
  const float *recurrent_to_input_weights_ptr, const float *recurrent_to_forget_weights_ptr,
  const float *recurrent_to_cell_weights_ptr, const float *recurrent_to_output_weights_ptr,
  const float *cell_to_input_weights_ptr, const float *cell_to_forget_weights_ptr,
  const float *cell_to_output_weights_ptr, const float *input_layer_norm_coefficients_ptr,
  const float *forget_layer_norm_coefficients_ptr, const float *cell_layer_norm_coefficients_ptr,
  const float *output_layer_norm_coefficients_ptr, const float *input_gate_bias_ptr,
  const float *forget_gate_bias_ptr, const float *cell_gate_bias_ptr,
  const float *output_gate_bias_ptr, const float *projection_weights_ptr,
  const float *projection_bias_ptr, const TfLiteLSTMParams *params, int n_batch, int n_cell,
  int n_input, int n_aux_input, int n_output, int output_batch_leading_dim, float *output_state_ptr,
  float *cell_state_ptr, float *scratch0, float *scratch1, float *scratch2, float *scratch3,
  float *output_ptr);

// Same as above but with quantized weight matrices. In detail:
// Input of size 'n_batch * n_input':
//   input_ptr
// Input of size 'n_batch * n_aux_input':
//   aux_input_ptr                     - optional (can be nullptr)
//
// LSTM weights:
// Quantized input weights of size 'n_cell * n_input':
//   input_to_input_weights            - optional
//   input_to_forget_weights
//   input_to_cell_weights
//   input_to_input_weights
// Quantized auxiliary input weights of size 'n_cell * n_aux_input':
//   aux_input_to_input_weights        - optional
//   aux_input_to_forget_weights       - optional
//   aux_input_to_cell_weights         - optional
//   aux_input_to_output_weights       - optional
// Quantized recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weights        - optional
//   recurrent_to_forget_weights
//   recurrent_to_cell_weights
//   recurrent_to_input_weights
// Quantized peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights             - optional
//   cell_to_cell_weights              - optional
//   cell_to_output_weights            - optional
// Quantized projection weights of size 'n_output * n_cell'
//   projection_weights_ptr            - optional
// Weight scales (scalars) for each of the weights above.
//   input_to_input_weights_scale      - optional
//   input_to_forget_weights_scale
//   input_to_cell_weights_scale
//   input_to_output_weights_scale
//   aux_input_to_input_weights_scale  - optional
//   aux_input_to_forget_weights_scale - optional
//   aux_input_to_cell_weights_scale   - optional
//   aux_input_to_output_weights_scale - optional
//   recurrent_to_input_weights_scale  - optional
//   recurrent_to_forget_weights_scale
//   recurrent_to_cell_weights_scale
//   recurrent_to_output_weights_scale
//   cell_to_input_weights_scale,
//   cell_to_forget_weights_scale,
//   cell_to_output_weights_scale,
//   projection_weights_scale          - optional
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr               - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Layer norm coefficients of size 'n_cell', representing diagonal matrices.
//   input_layer_norm_coefficients_ptr  - optional
//   forget_layer_norm_coefficients_ptr - optional
//   cell_layer_norm_coefficients_ptr   - optional
//   output_layer_norm_coefficients_ptr - optional
//
// Temporary pre-allocated storage for quantized values:
//   quantized_input_ptr (same size as input_ptr)
//   quantized_output_state_ptr (same size as output_state_ptr)
//   quantized_output_scratch (same size as cell_state_ptr)
// Temporary pre-allocated storage for recovered values:
//   recovered_cell_weights (same size as cell_to_*_weights)
//
// Outputs:
//   output_state_ptr - size 'n_batch * n_output'
//   cell_state_ptr   - size 'n_batch * n_cell'
//   output_ptr       - size 'n_batch * output_batch_leading_dim'
void LstmStepHybrid(
  const float *input_ptr, const int8_t *input_to_input_weights_ptr,
  const uint8_t *input_to_input_weights_ledger_ptr, float input_to_input_weights_scale,
  const int8_t *input_to_forget_weights_ptr, const uint8_t *input_to_forget_weights_ledger_ptr,
  float input_to_forget_weights_scale, const int8_t *input_to_cell_weights_ptr,
  const uint8_t *input_to_cell_weights_ledger_ptr, float input_to_cell_weights_scale,
  const int8_t *input_to_output_weights_ptr, const uint8_t *input_to_output_weights_ledger_ptr,
  float input_to_output_weights_scale, const float *aux_input_ptr,
  const int8_t *aux_input_to_input_weights_ptr, float aux_input_to_input_weights_scale,
  const int8_t *aux_input_to_forget_weights_ptr, float aux_input_to_forget_weights_scale,
  const int8_t *aux_input_to_cell_weights_ptr, float aux_input_to_cell_weights_scale,
  const int8_t *aux_input_to_output_weights_ptr, float aux_input_to_output_weights_scale,
  const int8_t *recurrent_to_input_weights_ptr,
  const uint8_t *recurrent_to_input_weights_ledger_ptr, float recurrent_to_input_weights_scale,
  const int8_t *recurrent_to_forget_weights_ptr,
  const uint8_t *recurrent_to_forget_weights_ledger_ptr, float recurrent_to_forget_weights_scale,
  const int8_t *recurrent_to_cell_weights_ptr, const uint8_t *recurrent_to_cell_weights_ledger_ptr,
  float recurrent_to_cell_weights_scale, const int8_t *recurrent_to_output_weights_ptr,
  const uint8_t *recurrent_to_output_weights_ledger_ptr, float recurrent_to_output_weights_scale,
  const int8_t *cell_to_input_weights_ptr, float cell_to_input_weights_scale,
  const int8_t *cell_to_forget_weights_ptr, float cell_to_forget_weights_scale,
  const int8_t *cell_to_output_weights_ptr, float cell_to_output_weights_scale,
  const float *input_layer_norm_coefficients_ptr, const float *forget_layer_norm_coefficients_ptr,
  const float *cell_layer_norm_coefficients_ptr, const float *output_layer_norm_coefficients_ptr,
  const float *input_gate_bias_ptr, const float *forget_gate_bias_ptr,
  const float *cell_gate_bias_ptr, const float *output_gate_bias_ptr,
  const int8_t *projection_weights_ptr, const uint8_t *projection_weights_ledger_ptr,
  float projection_weights_scale, const float *projection_bias_ptr, const TfLiteLSTMParams *params,
  int n_batch, int n_cell, int n_input, int n_aux_input, int n_output, int output_batch_leading_dim,
  float *scratch0, float *scratch1, float *scratch2, float *scratch3, float *input_sf,
  float *aux_input_sf, float *output_state_sf, float *scaling_factors_scratch,
  float *recovered_cell_weights, int8_t *quantized_input_ptr, int8_t *quantized_aux_input_ptr,
  int8_t *quantized_output_state_ptr, int8_t *quantized_output_scratch, float *output_state_ptr,
  float *cell_state_ptr, int32_t *accum_scratch_ptr, float *output_ptr, int32_t *input_zp,
  int32_t *aux_input_zp, int32_t *output_state_zp, int32_t *row_sums, int row_sums_size,
  bool *compute_row_sums, bool asymmetric_quantize_inputs);

} // namespace eval_lstm
} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_EVAL_LSTM_HELPER_H
