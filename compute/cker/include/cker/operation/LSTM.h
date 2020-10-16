/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_UNIDIRECTIONALSEQUENCELSTM_H__
#define __NNFW_CKER_UNIDIRECTIONALSEQUENCELSTM_H__

#include "cker/TensorUtils.h"
#include "cker/Types.h"

namespace nnfw
{
namespace cker
{

// LINT.IfChange
// Calculates a single LSTM gate.
//
// Implements the following formula: (* is matrix multiply)
//   gate = activate(W_input    * input + W_aux       * aux_input   +
//                   W_peephole * cell  + W_recurrent * prev_output + bias)
// with layer norm:
//   gate = activate(W_norm * normalize(...) + bias) // not adding bias inside
//
// Activation is sigmoid except for the "cell" gate (configurable, usually tanh)
//
// Parameters:
// Input vectors (to LSTM):    | Size:                | Optional?
//   input                     | n_input              |
//   aux_input                 | n_aux_input          | y (bidir LSTM)
// Input vectors (persistent states):
//   output_state              | n_output             |
//   cell_state                | n_cell               |
// 'Constant' inputs:
//   input_to_gate_weights     | n_cell * n_input     |
//   aux_input_to_gate_weights | n_cell * n_aux_input | y (bidir LSTM)
//   recurrent_to_gate_weights | n_cell * n_output    |
//   cell_to_gate_weights      | n_cell               | y (peephole)
//   gate_bias                 | n_cell               |
//   layer_norm_coefficients   | n_cell               | y (layer norm)
// Output vector:
//   gate                      | n_cell               |
// Scalar parameters:
//   n_batch                                    - batch size / number of vectors
//   n_input, n_aux_input, n_output, n_cell     - size of vectors.
//   activation                                 - activation to use.
//   is_input_all_zeros, is_aux_input_all_zeros - if input vectors are all zero.
//   use_layer_norm                             - if doing layer norm LSTM.
inline void CalculateLstmGateFloat(const float *input, const float *input_to_gate_weights,
                                   const float *aux_input, const float *aux_input_to_gate_weights,
                                   const float *output_state,
                                   const float *recurrent_to_gate_weights, const float *cell_state,
                                   const float *cell_to_gate_weights,
                                   const float *layer_norm_coefficients, const float *gate_bias,
                                   const int n_batch, const int n_input, const int n_aux_input,
                                   const int n_output, const int n_cell,
                                   const FusedActivationFunctionType activation, float *gate,
                                   const bool is_input_all_zeros, const bool is_aux_input_all_zeros)
{
  const bool use_peephole = (cell_to_gate_weights != nullptr);
  const bool use_layer_norm = (layer_norm_coefficients != nullptr);

  // Initialize scratch buffers with bias for regular lstm or initialize with
  // zero for layer norm lstm.
  if (use_layer_norm)
  {
    std::fill_n(gate, n_cell * n_batch, 0.0f);
  }
  else
  {
    VectorBatchVectorAssign(gate_bias, n_cell, n_batch, gate);
  }
  // For each batch and cell: compute input_weight * input.
  // Skip if input is all zeros.
  if (!is_input_all_zeros)
  {
    MatrixBatchVectorMultiplyAccumulate(input_to_gate_weights, n_cell, n_input, input, n_batch,
                                        gate, /*result_stride=*/1);
  }
  // For each batch and cell: compute aux_input_weight * aux_input.
  // Skip if auxiliary input is not available or all zeros.
  if (!is_aux_input_all_zeros)
  {
    MatrixBatchVectorMultiplyAccumulate(aux_input_to_gate_weights, n_cell, n_aux_input, aux_input,
                                        n_batch, gate, /*result_stride=*/1);
  }
  // For each batch and cell: compute recurrent_weight * output_state.
  MatrixBatchVectorMultiplyAccumulate(recurrent_to_gate_weights, n_cell, n_output, output_state,
                                      n_batch, gate, /*result_stride=*/1);
  // For each batch and cell: compute cell_weight .* cell_state (peephole LSTM)
  if (use_peephole)
  {
    VectorBatchVectorCwiseProductAccumulate(cell_to_gate_weights, n_cell, cell_state, n_batch,
                                            gate);
  }
  // Do layer normalization (if layer norm LSTM)
  if (use_layer_norm)
  {
    MeanStddevNormalization(gate, gate, n_cell, n_batch);
    VectorBatchVectorCwiseProduct(layer_norm_coefficients, n_cell, gate, n_batch, gate);
    VectorBatchVectorAdd(gate_bias, n_cell, n_batch, gate);
  }
  // Apply activation
  ApplyActivationToVector(gate, n_batch * n_cell, activation, gate);
}

// Updates the LSTM cell state, used by both float and hybrid LSTM versions.
//
// Implements the following formula:
//   cell_state_new = clip(forget_gate * cell_state + input_gate * cell_gate)
//
// With CIFG LSTM, input gate is replaced by (1-forget_gate).
//
// Parameters:
//  - n_batch, n_cell: sizes of vectors
//  - cell_state: input/output vector, size n_batch*n_cell
//  - input_gate: input vector, size n_batch*n_cell.
//  - forget_gate: input/scratch vector, size n_batch*n_cell, modified with CIFG
//  - cell_gate: input vector, size n_batch*n_cell.
//  - use_cifg: use 1-forget_gate instead of input_gate.
//  - clip: if > 0, clip the resulting cell state to [-clip, +clip].
void UpdateLstmCellFloat(int n_batch, int n_cell, float *cell_state, const float *input_gate,
                         float *forget_gate, const float *cell_gate, bool use_cifg, float clip)
{
  // Define variable for 4th argument to avoid warning
  // Compiler warning: passing argument 4 to restrict-qualified parameter aliases with argument 2
  const float *cwise_product_rhs = cell_state;
  VectorVectorCwiseProduct(forget_gate, cwise_product_rhs, n_batch * n_cell, cell_state);

  if (use_cifg)
  {
    // With CIFG, input_gate = 1-forget_gate. Use the forget_gate array as
    // scratch, as input_gate array is not allocated in this case. (Be careful
    // not to write to the scratch before reading the forget gate data.)
    float *scratch = forget_gate;
    Sub1Vector(forget_gate, n_batch * n_cell, scratch);
    VectorVectorCwiseProductAccumulate(cell_gate, scratch, n_batch * n_cell, cell_state);
  }
  else
  {
    VectorVectorCwiseProductAccumulate(cell_gate, input_gate, n_batch * n_cell, cell_state);
  }
  if (clip > 0.0f)
  {
    CwiseClipping(cell_state, n_batch * n_cell, clip);
  }
}

// Calculates the output state tensor of an LSTM step.
//
// Implements the following formula:
//   output_no_projection = output_gate .* activate(cell_state)
//     (elementwise vector product)
// If no projection is used:
//   output = output_state = output_no_projection
// With projection:
//   output = output_state = clip(W*output_no_projection + bias)
//
// Output might not have a different 'stride' than n_batch, so we need to copy.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - projection_weights, projection_weights_scale, projection_bias:
//      constant inputs, describing projection matrix and bias.
//  - proj_clip: if > 0, clip the output of the projection.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - scratch: scratch area, size n_batch*n_cell.
void CalculateLstmOutputFloat(int n_batch, int n_cell, int n_output, const float *cell_state,
                              const float *output_gate, FusedActivationFunctionType activation,
                              const float *projection_weights, const float *projection_bias,
                              const float proj_clip, float *output_state, float *scratch)
{
  ApplyActivationToVector(cell_state, n_batch * n_cell, activation, scratch);

  // Define variable for 4th argument to avoid warning
  // Compiler warning: passing argument 4 to restrict-qualified parameter aliases with argument 2
  const float *cwise_product_rhs = scratch;
  VectorVectorCwiseProduct(output_gate, cwise_product_rhs, n_batch * n_cell, scratch);

  const bool use_projection = (projection_weights != nullptr);
  const bool use_projection_bias = (projection_bias != nullptr);

  if (use_projection)
  {
    if (use_projection_bias)
    {
      VectorBatchVectorAssign(projection_bias, n_output, n_batch, output_state);
    }
    else
    {
      std::fill_n(output_state, n_batch * n_output, 0.0f);
    }
    MatrixBatchVectorMultiplyAccumulate(projection_weights, n_output, n_cell, scratch, n_batch,
                                        output_state, /*result_stride=*/1);
    if (proj_clip > 0.0f)
    {
      CwiseClipping(output_state, n_batch * n_output, proj_clip);
    }
  }
  else
  {
    std::copy_n(scratch, n_batch * n_output, output_state);
  }
}

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
// LINT.IfChange
inline void LstmStepFloat(
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
    const float *projection_bias_ptr, const LSTMParams *params, int n_batch, int n_cell,
    int n_input, int n_aux_input, int n_output, int output_batch_leading_dim,
    float *output_state_ptr, float *cell_state_ptr, float *scratch0, float *scratch1,
    float *scratch2, float *scratch3, float *output_ptr)
{
  // Since we have already checked that weights are all there or none, we can
  // check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);

  // Make named scratch buffers.
  float *input_gate_scratch = scratch0;
  float *forget_gate_scratch = scratch1;
  float *cell_gate_scratch = scratch2;
  float *output_gate_scratch = scratch3;

  // Check if inputs are all zeros so we can skip some computations.
  const bool is_input_all_zeros = IsZeroVector(input_ptr, n_batch * n_input);
  const bool is_aux_input_all_zeros =
      (aux_input_ptr == nullptr || IsZeroVector(aux_input_ptr, n_batch * n_aux_input));
  if (!use_cifg)
  {
    // Calculate the input gate. (If not CIFG.)
    CalculateLstmGateFloat(input_ptr, input_to_input_weights_ptr, aux_input_ptr,
                           aux_input_to_input_weights_ptr, output_state_ptr,
                           recurrent_to_input_weights_ptr, cell_state_ptr,
                           cell_to_input_weights_ptr, input_layer_norm_coefficients_ptr,
                           input_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
                           /*activation=kTfLiteActSigmoid*/ FusedActivationFunctionType::kSigmoid,
                           input_gate_scratch, is_input_all_zeros, is_aux_input_all_zeros);
  }
  // Calculate the forget gate.
  CalculateLstmGateFloat(input_ptr, input_to_forget_weights_ptr, aux_input_ptr,
                         aux_input_to_forget_weights_ptr, output_state_ptr,
                         recurrent_to_forget_weights_ptr, cell_state_ptr,
                         cell_to_forget_weights_ptr, forget_layer_norm_coefficients_ptr,
                         forget_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
                         /*activation=kTfLiteActSigmoid*/ FusedActivationFunctionType::kSigmoid,
                         forget_gate_scratch, is_input_all_zeros, is_aux_input_all_zeros);
  // Calculate the cell update gate.
  CalculateLstmGateFloat(
      input_ptr, input_to_cell_weights_ptr, aux_input_ptr, aux_input_to_cell_weights_ptr,
      output_state_ptr, recurrent_to_cell_weights_ptr, /*cell_state=*/nullptr,
      /*cell_to_gate_weights=*/nullptr, cell_layer_norm_coefficients_ptr, cell_gate_bias_ptr,
      n_batch, n_input, n_aux_input, n_output, n_cell, params->activation, cell_gate_scratch,
      is_input_all_zeros, is_aux_input_all_zeros);
  // Update the cell state.
  UpdateLstmCellFloat(n_batch, n_cell, cell_state_ptr, input_gate_scratch, forget_gate_scratch,
                      cell_gate_scratch, use_cifg, params->cell_clip);
  // Calculate output gate.
  CalculateLstmGateFloat(input_ptr, input_to_output_weights_ptr, aux_input_ptr,
                         aux_input_to_output_weights_ptr, output_state_ptr,
                         recurrent_to_output_weights_ptr, cell_state_ptr,
                         cell_to_output_weights_ptr, output_layer_norm_coefficients_ptr,
                         output_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
                         /*activation=kTfLiteActSigmoid*/ FusedActivationFunctionType::kSigmoid,
                         output_gate_scratch, is_input_all_zeros, is_aux_input_all_zeros);
  // Update the output state.
  CalculateLstmOutputFloat(n_batch, n_cell, n_output, cell_state_ptr, output_gate_scratch,
                           params->activation, projection_weights_ptr, projection_bias_ptr,
                           params->proj_clip, output_state_ptr, scratch2);
  // Copy output state to the output. Note that the output's rows may not be
  // contiguous (output_batch_leading_dim != n_output).
  for (int b = 0; b < n_batch; b++)
  {
    std::copy_n(output_state_ptr + b * n_output, n_output,
                output_ptr + b * output_batch_leading_dim);
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_UNIDIRECTIONALSEQUENCELSTM_H__
