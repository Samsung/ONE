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

#include "EvalLSTMHelper.h"
#include <vector>

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
namespace
{
void PortableMeanStddevNormalization(const float *input_vector, float *output_vector, int v_size,
                                     int n_batch)
{
  for (int batch = 0; batch < n_batch; ++batch)
  {
    float sum = 0.0f;
    for (int i = 0; i < v_size; ++i)
    {
      sum += input_vector[i];
    }
    const float mean = sum / v_size;
    float sum_diff_sq = 0.0f;
    for (int i = 0; i < v_size; ++i)
    {
      const float diff = input_vector[i] - mean;
      sum_diff_sq += diff * diff;
    }
    const float variance = sum_diff_sq / v_size;
    constexpr float kNormalizationConstant = 1e-8f;
    const float stddev_inv = 1.0f / std::sqrt(variance + kNormalizationConstant);
    for (int i = 0; i < v_size; ++i)
    {
      output_vector[i] = (input_vector[i] - mean) * stddev_inv;
    }
    input_vector += v_size;
    output_vector += v_size;
  }
}

inline void CalculateLstmGateFloat(const float *input, const float *input_to_gate_weights,
                                   const float *aux_input, const float *aux_input_to_gate_weights,
                                   const float *output_state,
                                   const float *recurrent_to_gate_weights, const float *cell_state,
                                   const float *cell_to_gate_weights,
                                   const float *layer_norm_coefficients, const float *gate_bias,
                                   const int n_batch, const int n_input, const int n_aux_input,
                                   const int n_output, const int n_cell,
                                   const TfLiteFusedActivation activation, float *gate,
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
    tflite::tensor_utils::VectorBatchVectorAssign(gate_bias, n_cell, n_batch, gate);
  }
  // For each batch and cell: compute input_weight * input.
  // Skip if input is all zeros.
  if (!is_input_all_zeros)
  {
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(input_to_gate_weights, n_cell,
                                                              n_input, input, n_batch, gate);
  }
  // For each batch and cell: compute aux_input_weight * aux_input.
  // Skip if auxiliary input is not available or all zeros.
  if (!is_aux_input_all_zeros)
  {
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      aux_input_to_gate_weights, n_cell, n_aux_input, aux_input, n_batch, gate);
  }
  // For each batch and cell: compute recurrent_weight * output_state.
  tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(recurrent_to_gate_weights, n_cell,
                                                            n_output, output_state, n_batch, gate);
  // For each batch and cell: compute cell_weight .* cell_state (peephole LSTM)
  if (use_peephole)
  {
    tflite::tensor_utils::VectorBatchVectorCwiseProductAccumulate(cell_to_gate_weights, n_cell,
                                                                  cell_state, n_batch, gate);
  }
  // Do layer normalization (if layer norm LSTM)
  if (use_layer_norm)
  {
    // tflite::tensor_utils::MeanStddevNormalization(gate, gate, n_cell, n_batch);
    PortableMeanStddevNormalization(gate, gate, n_cell, n_batch);
    tflite::tensor_utils::VectorBatchVectorCwiseProduct(layer_norm_coefficients, n_cell, gate,
                                                        n_batch, gate);
    tflite::tensor_utils::VectorBatchVectorAdd(gate_bias, n_cell, n_batch, gate);
  }
  // Apply activation
  tflite::tensor_utils::ApplyActivationToVector(gate, n_batch * n_cell, activation, gate);
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
                              const float *output_gate, TfLiteFusedActivation activation,
                              const float *projection_weights, const float *projection_bias,
                              const float proj_clip, float *output_state, float *scratch)
{
  tflite::tensor_utils::ApplyActivationToVector(cell_state, n_batch * n_cell, activation, scratch);
  tflite::tensor_utils::VectorVectorCwiseProduct(output_gate, scratch, n_batch * n_cell, scratch);

  const bool use_projection = (projection_weights != nullptr);
  const bool use_projection_bias = (projection_bias != nullptr);

  if (use_projection)
  {
    if (use_projection_bias)
    {
      tflite::tensor_utils::VectorBatchVectorAssign(projection_bias, n_output, n_batch,
                                                    output_state);
    }
    else
    {
      std::fill_n(output_state, n_batch * n_output, 0.0f);
    }
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(projection_weights, n_output, n_cell,
                                                              scratch, n_batch, output_state);
    if (proj_clip > 0.0f)
    {
      tflite::tensor_utils::CwiseClipping(output_state, n_batch * n_output, proj_clip);
    }
  }
  else
  {
    std::copy_n(scratch, n_batch * n_output, output_state);
  }
}

// Calculates a single LSTM gate, hybrid version.
// Implements the same functionality as CalculateLstmGateFloat.
void CalculateLstmGateHybrid(
  // Input and weights
  const int8_t *input, const float *input_sf, const int32_t *input_zp,
  const int8_t *input_to_gate_weights, const uint8_t *input_to_gate_weights_ledger,
  const float input_to_gate_weights_scale, int32_t *input_to_gate_row_sums,
  // Aux input and weights
  const int8_t *aux_input, const float *aux_input_sf, const int32_t *aux_input_zp,
  const int8_t *aux_input_to_gate_weights, const float aux_input_to_gate_weights_scale,
  int32_t *aux_input_to_gate_row_sums,
  // Output state and weights
  const int8_t *output_state, const float *output_state_sf, const int32_t *output_state_zp,
  const int8_t *recurrent_to_gate_weights, const uint8_t *recurrent_to_gate_weights_ledger,
  const float recurrent_to_gate_weights_scale, int32_t *recurrent_to_gate_row_sums,
  // Cell state and weights (peephole LSTM)
  const float *cell_state, const int8_t *cell_to_gate_weights,
  const float cell_to_gate_weights_scale,
  // Layer normalization coefficients (layer norm LSTM) + gate bias
  const float *layer_norm_coefficients, const float *gate_bias,
  // Array sizes
  const int n_batch, const int n_input, const int n_aux_input, const int n_output, const int n_cell,
  const TfLiteFusedActivation activation,
  // Output
  float *gate,
  // Parameters for performance optimizations
  const bool is_input_all_zeros, const bool is_aux_input_all_zeros,
  const bool is_output_state_all_zeros, bool *compute_row_sums,
  // Scratch arrays
  float *scratch0,       // size: n_batch
  float *scratch1,       // size: n_cell, only used if peephole LSTM
  int32_t *accum_scratch // For MatrixBatchVectorMultiplyAccumulate
)
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
    tflite::tensor_utils::VectorBatchVectorAssign(gate_bias, n_cell, n_batch, gate);
  }
  // For each batch and cell: compute input_weight * input.
  // Skip if input is all zeros.
  if (!is_input_all_zeros)
  {
    if (input_to_gate_weights_ledger != nullptr)
    {
      std::vector<float> scales(n_batch);
      for (int i = 0; i < n_batch; i++)
      {
        scales[i] = input_to_gate_weights_scale * input_sf[i];
      }
      tflite::tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
        input_to_gate_weights, input_to_gate_weights_ledger, n_cell, n_input, input, scales.data(),
        n_batch, gate);
    }
    else
    {
      tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        input_to_gate_weights, n_cell, n_input, input, input_to_gate_weights_scale, input_sf,
        n_batch, gate,
        /*per_channel_scale=*/nullptr, input_zp, accum_scratch, input_to_gate_row_sums,
        compute_row_sums, scratch0, /*context=*/nullptr);
    }
  }
  // For each batch and cell: compute aux_input_weight * aux_input.
  // Skip if auxiliary input is not available or all zeros.
  if (!is_aux_input_all_zeros)
  {
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      aux_input_to_gate_weights, n_cell, n_aux_input, aux_input, aux_input_to_gate_weights_scale,
      aux_input_sf, n_batch, gate,
      /*per_channel_scale=*/nullptr, aux_input_zp, accum_scratch, aux_input_to_gate_row_sums,
      compute_row_sums, scratch0, /*context=*/nullptr);
  }
  // For each batch and cell: compute recurrent_weight * output_state.
  // Skip if output state is all zeros.
  if (!is_output_state_all_zeros)
  {
    if (recurrent_to_gate_weights_ledger != nullptr)
    {
      std::vector<float> scales(n_batch);
      for (int i = 0; i < n_batch; i++)
      {
        scales[i] = recurrent_to_gate_weights_scale * input_sf[i];
      }
      tflite::tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
        recurrent_to_gate_weights, recurrent_to_gate_weights_ledger, n_cell, n_output, output_state,
        scales.data(), n_batch, gate);
    }
    else
    {
      tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        recurrent_to_gate_weights, n_cell, n_output, output_state, recurrent_to_gate_weights_scale,
        output_state_sf, n_batch, gate,
        /*per_channel_scale=*/nullptr, output_state_zp, accum_scratch, recurrent_to_gate_row_sums,
        compute_row_sums, scratch0, nullptr);
    }
  }
  // For each batch and cell: compute cell_weight .* cell_state (peephole LSTM)
  if (use_peephole)
  {
    float *recovered_cell_weights = scratch1;
    tflite::tensor_utils::VectorScalarMultiply(cell_to_gate_weights, n_cell,
                                               cell_to_gate_weights_scale, recovered_cell_weights);
    tflite::tensor_utils::VectorBatchVectorCwiseProductAccumulate(recovered_cell_weights, n_cell,
                                                                  cell_state, n_batch, gate);
  }
  // Do layer normalization (if layer norm LSTM)
  if (use_layer_norm)
  {
    // tflite::tensor_utils::MeanStddevNormalization(gate, gate, n_cell, n_batch);
    PortableMeanStddevNormalization(gate, gate, n_cell, n_batch);
    tflite::tensor_utils::VectorBatchVectorCwiseProduct(layer_norm_coefficients, n_cell, gate,
                                                        n_batch, gate);
    tflite::tensor_utils::VectorBatchVectorAdd(gate_bias, n_cell, n_batch, gate);
  }
  // Apply activation
  tflite::tensor_utils::ApplyActivationToVector(gate, n_cell * n_batch, activation, gate);
}

void ComputeRowSums(
  int32_t *input_to_input_row_sums, int32_t *input_to_forget_row_sums,
  int32_t *input_to_cell_row_sums, int32_t *input_to_output_row_sums,
  int32_t *aux_input_to_input_row_sums, int32_t *aux_input_to_forget_row_sums,
  int32_t *aux_input_to_cell_row_sums, int32_t *aux_input_to_output_row_sums,
  int32_t *recurrent_to_input_row_sums, int32_t *recurrent_to_forget_row_sums,
  int32_t *recurrent_to_cell_row_sums, int32_t *recurrent_to_output_row_sums,
  int32_t *projection_weights_row_sums, int32_t *row_sums, int n_cell, int n_input, int n_aux_input,
  int n_output, const int8_t *input_to_input_weights_ptr, const int8_t *input_to_forget_weights_ptr,
  const int8_t *input_to_cell_weights_ptr, const int8_t *input_to_output_weights_ptr,
  const int8_t *aux_input_to_input_weights_ptr, const int8_t *aux_input_to_forget_weights_ptr,
  const int8_t *aux_input_to_cell_weights_ptr, const int8_t *aux_input_to_output_weights_ptr,
  const int8_t *recurrent_to_input_weights_ptr, const int8_t *recurrent_to_forget_weights_ptr,
  const int8_t *recurrent_to_cell_weights_ptr, const int8_t *recurrent_to_output_weights_ptr,
  const int8_t *projection_weights_ptr, bool use_cifg, const float *aux_input_ptr)
{
  // Compute the row sums for dequantization
  if (!use_cifg)
  {
    tflite::tensor_utils::ReductionSumVector(input_to_input_weights_ptr, input_to_input_row_sums,
                                             n_cell, n_input);
  }
  tflite::tensor_utils::ReductionSumVector(input_to_forget_weights_ptr, input_to_forget_row_sums,
                                           n_cell, n_input);
  tflite::tensor_utils::ReductionSumVector(input_to_cell_weights_ptr, input_to_cell_row_sums,
                                           n_cell, n_input);
  tflite::tensor_utils::ReductionSumVector(input_to_output_weights_ptr, input_to_output_row_sums,
                                           n_cell, n_input);

  if (aux_input_ptr)
  {
    if (!use_cifg)
    {
      tflite::tensor_utils::ReductionSumVector(aux_input_to_input_weights_ptr,
                                               aux_input_to_input_row_sums, n_cell, n_aux_input);
    }
    tflite::tensor_utils::ReductionSumVector(aux_input_to_forget_weights_ptr,
                                             aux_input_to_forget_row_sums, n_cell, n_aux_input);
    tflite::tensor_utils::ReductionSumVector(aux_input_to_cell_weights_ptr,
                                             aux_input_to_cell_row_sums, n_cell, n_aux_input);
    tflite::tensor_utils::ReductionSumVector(aux_input_to_output_weights_ptr,
                                             aux_input_to_output_row_sums, n_cell, n_aux_input);
  }
  if (!use_cifg)
  {
    tflite::tensor_utils::ReductionSumVector(recurrent_to_input_weights_ptr,
                                             recurrent_to_input_row_sums, n_cell, n_output);
  }
  tflite::tensor_utils::ReductionSumVector(recurrent_to_forget_weights_ptr,
                                           recurrent_to_forget_row_sums, n_cell, n_output);
  tflite::tensor_utils::ReductionSumVector(recurrent_to_cell_weights_ptr,
                                           recurrent_to_cell_row_sums, n_cell, n_output);
  tflite::tensor_utils::ReductionSumVector(recurrent_to_output_weights_ptr,
                                           recurrent_to_output_row_sums, n_cell, n_output);

  if (projection_weights_ptr != nullptr)
  {
    tflite::tensor_utils::ReductionSumVector(projection_weights_ptr, projection_weights_row_sums,
                                             n_output, n_cell);
  }
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
  tflite::tensor_utils::VectorVectorCwiseProduct(forget_gate, cell_state, n_batch * n_cell,
                                                 cell_state);

  if (use_cifg)
  {
    // With CIFG, input_gate = 1-forget_gate. Use the forget_gate array as
    // scratch, as input_gate array is not allocated in this case. (Be careful
    // not to write to the scratch before reading the forget gate data.)
    float *scratch = forget_gate;
    tflite::tensor_utils::Sub1Vector(forget_gate, n_batch * n_cell, scratch);
    tflite::tensor_utils::VectorVectorCwiseProductAccumulate(cell_gate, scratch, n_batch * n_cell,
                                                             cell_state);
  }
  else
  {
    tflite::tensor_utils::VectorVectorCwiseProductAccumulate(cell_gate, input_gate,
                                                             n_batch * n_cell, cell_state);
  }
  if (clip > 0.0f)
  {
    tflite::tensor_utils::CwiseClipping(cell_state, n_batch * n_cell, clip);
  }
}

// Calculates the output state tensor of an LSTM step. See Float version too.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - projection_weights, projection_weights_scale, projection_bias:
//      constant inputs, describing projection matrix and bias.
//  - proj_clip: if > 0, clip the output of the projection.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - asymmetric_quantize_inputs: parameter to control quantization.
//  - projection_weights_row_sums, compute_row_sums, context: Data for optimized
//      MatrixBatchVectorMultiplyAccumulate.
//  - scratch0: scratch area of size n_batch*n_cell
//  - scratch1: scratch area of size n_batch*n_cell
//  - scratch2: scratch area of size n_batch
//  - scratch3: scratch area of size n_batch
//  - scratch4: scratch area used by MatrixBatchVectorMultiplyAccumulate
void CalculateLstmOutputHybrid(
  int n_batch, int n_cell, int n_output, const float *cell_state, const float *output_gate,
  TfLiteFusedActivation activation, const int8_t *projection_weights,
  const uint8_t *projection_weights_ledger, float projection_weights_scale,
  const float *projection_bias, const float proj_clip, float *output_state,
  bool asymmetric_quantize_inputs, int32_t *projection_weights_row_sums, bool *compute_row_sums,
  float *scratch0, int8_t *scratch1, float *scratch2, int32_t *scratch3, int32_t *scratch4)
{
  tflite::tensor_utils::ApplyActivationToVector(cell_state, n_batch * n_cell, activation, scratch0);
  tflite::tensor_utils::VectorVectorCwiseProduct(output_gate, scratch0, n_batch * n_cell, scratch0);

  const bool use_projection = (projection_weights != nullptr);
  const bool use_projection_bias = (projection_bias != nullptr);

  if (use_projection)
  {
    if (use_projection_bias)
    {
      tflite::tensor_utils::VectorBatchVectorAssign(projection_bias, n_output, n_batch,
                                                    output_state);
    }
    else
    {
      std::fill_n(output_state, n_batch * n_output, 0.0f);
    }
    if (!tflite::tensor_utils::IsZeroVector(scratch0, n_batch * n_cell))
    {
      // Save quantization and matmul computation for all zero output.
      tflite::tensor_utils::BatchQuantizeFloats(scratch0, n_batch, n_cell, scratch1, scratch2,
                                                scratch3, asymmetric_quantize_inputs);
      if (projection_weights_ledger != nullptr)
      {
        std::vector<float> scales(n_batch);
        for (int i = 0; i < n_batch; i++)
        {
          scales[i] = projection_weights_scale * scratch2[i];
        }
        tflite::tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate(
          projection_weights, projection_weights_ledger, n_output, n_cell, scratch1, scales.data(),
          n_batch, output_state);
      }
      else
      {
        tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
          projection_weights, n_output, n_cell, scratch1, projection_weights_scale, scratch2,
          n_batch, output_state,
          /*per_channel_scale=*/nullptr, scratch3, scratch4, projection_weights_row_sums,
          compute_row_sums, scratch2, nullptr);
      }
    }
    if (proj_clip > 0.0f)
    {
      tflite::tensor_utils::CwiseClipping(output_state, n_batch * n_output, proj_clip);
    }
  }
  else
  {
    std::copy_n(scratch0, n_batch * n_output, output_state);
  }
}
} // namespace

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
  float *output_ptr)
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
  const bool is_input_all_zeros = tflite::tensor_utils::IsZeroVector(input_ptr, n_batch * n_input);
  const bool is_aux_input_all_zeros =
    (aux_input_ptr == nullptr ||
     tflite::tensor_utils::IsZeroVector(aux_input_ptr, n_batch * n_aux_input));
  if (!use_cifg)
  {
    // Calculate the input gate. (If not CIFG.)
    CalculateLstmGateFloat(input_ptr, input_to_input_weights_ptr, aux_input_ptr,
                           aux_input_to_input_weights_ptr, output_state_ptr,
                           recurrent_to_input_weights_ptr, cell_state_ptr,
                           cell_to_input_weights_ptr, input_layer_norm_coefficients_ptr,
                           input_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
                           /*activation=*/kTfLiteActSigmoid, input_gate_scratch, is_input_all_zeros,
                           is_aux_input_all_zeros);
  }
  // Calculate the forget gate.
  CalculateLstmGateFloat(input_ptr, input_to_forget_weights_ptr, aux_input_ptr,
                         aux_input_to_forget_weights_ptr, output_state_ptr,
                         recurrent_to_forget_weights_ptr, cell_state_ptr,
                         cell_to_forget_weights_ptr, forget_layer_norm_coefficients_ptr,
                         forget_gate_bias_ptr, n_batch, n_input, n_aux_input, n_output, n_cell,
                         /*activation=*/kTfLiteActSigmoid, forget_gate_scratch, is_input_all_zeros,
                         is_aux_input_all_zeros);
  // Calculate the cell update gate.
  CalculateLstmGateFloat(
    input_ptr, input_to_cell_weights_ptr, aux_input_ptr, aux_input_to_cell_weights_ptr,
    output_state_ptr, recurrent_to_cell_weights_ptr, /*cell_state=*/nullptr,
    /*cell_to_gate_weights=*/nullptr, cell_layer_norm_coefficients_ptr, cell_gate_bias_ptr, n_batch,
    n_input, n_aux_input, n_output, n_cell, params->activation, cell_gate_scratch,
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
                         /*activation=*/kTfLiteActSigmoid, output_gate_scratch, is_input_all_zeros,
                         is_aux_input_all_zeros);
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
  bool *compute_row_sums, bool asymmetric_quantize_inputs)
{
  // Since we have already checked that weights are all there or none, we
  // can check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weights_ptr == nullptr);
  // Make named scratch buffers for the different gates.
  float *input_gate_scratch = scratch0;
  float *forget_gate_scratch = scratch1;
  float *cell_gate_scratch = scratch2;
  float *output_gate_scratch = scratch3;

  int32_t *input_to_input_row_sums = nullptr;
  int32_t *input_to_forget_row_sums = nullptr;
  int32_t *input_to_cell_row_sums = nullptr;
  int32_t *input_to_output_row_sums = nullptr;
  int32_t *aux_input_to_input_row_sums = nullptr;
  int32_t *aux_input_to_forget_row_sums = nullptr;
  int32_t *aux_input_to_cell_row_sums = nullptr;
  int32_t *aux_input_to_output_row_sums = nullptr;
  int32_t *recurrent_to_input_row_sums = nullptr;
  int32_t *recurrent_to_forget_row_sums = nullptr;
  int32_t *recurrent_to_cell_row_sums = nullptr;
  int32_t *recurrent_to_output_row_sums = nullptr;
  int32_t *projection_weights_row_sums = nullptr;

  if (asymmetric_quantize_inputs)
  {
    int num_row_sums = use_cifg ? 6 : 8;
    if (aux_input_ptr != nullptr)
    {
      num_row_sums += use_cifg ? 3 : 4;
    }
    if (projection_weights_ptr != nullptr)
    {
      num_row_sums += ceil(static_cast<float>(n_output) / n_cell);
    }
    input_to_input_row_sums = row_sums;
    input_to_forget_row_sums =
      use_cifg ? input_to_input_row_sums : input_to_input_row_sums + n_cell;
    input_to_cell_row_sums = input_to_forget_row_sums + n_cell;
    input_to_output_row_sums = input_to_cell_row_sums + n_cell;
    if (aux_input_ptr != nullptr)
    {
      aux_input_to_input_row_sums = input_to_output_row_sums + n_cell;
      aux_input_to_forget_row_sums =
        use_cifg ? aux_input_to_input_row_sums : aux_input_to_input_row_sums + n_cell;
      aux_input_to_cell_row_sums = aux_input_to_forget_row_sums + n_cell;
      aux_input_to_output_row_sums = aux_input_to_cell_row_sums + n_cell;
    }
    recurrent_to_input_row_sums =
      aux_input_ptr ? aux_input_to_output_row_sums + n_cell : input_to_output_row_sums + n_cell;
    recurrent_to_forget_row_sums =
      use_cifg ? recurrent_to_input_row_sums : recurrent_to_input_row_sums + n_cell;
    recurrent_to_cell_row_sums = recurrent_to_forget_row_sums + n_cell;
    recurrent_to_output_row_sums = recurrent_to_cell_row_sums + n_cell;
    if (projection_weights_ptr != nullptr)
    {
      projection_weights_row_sums = recurrent_to_output_row_sums + n_cell;
    }
    if (*compute_row_sums)
    {
      ComputeRowSums(
        input_to_input_row_sums, input_to_forget_row_sums, input_to_cell_row_sums,
        input_to_output_row_sums, aux_input_to_input_row_sums, aux_input_to_forget_row_sums,
        aux_input_to_cell_row_sums, aux_input_to_output_row_sums, recurrent_to_input_row_sums,
        recurrent_to_forget_row_sums, recurrent_to_cell_row_sums, recurrent_to_output_row_sums,
        projection_weights_row_sums, row_sums, n_cell, n_input, n_aux_input, n_output,
        input_to_input_weights_ptr, input_to_forget_weights_ptr, input_to_cell_weights_ptr,
        input_to_output_weights_ptr, aux_input_to_input_weights_ptr,
        aux_input_to_forget_weights_ptr, aux_input_to_cell_weights_ptr,
        aux_input_to_output_weights_ptr, recurrent_to_input_weights_ptr,
        recurrent_to_forget_weights_ptr, recurrent_to_cell_weights_ptr,
        recurrent_to_output_weights_ptr, projection_weights_ptr, use_cifg, aux_input_ptr);
      *compute_row_sums = false;
    }
  }

  // Check if inputs are all zeros so we can skip some computations.
  const bool is_input_all_zeros = tflite::tensor_utils::IsZeroVector(input_ptr, n_batch * n_input);
  const bool is_aux_input_all_zeros =
    (aux_input_ptr == nullptr ||
     tflite::tensor_utils::IsZeroVector(aux_input_ptr, n_batch * n_aux_input));
  const bool is_output_state_all_zeros =
    tflite::tensor_utils::IsZeroVector(output_state_ptr, n_batch * n_output);
  // Quantize inputs.
  if (!is_input_all_zeros)
  {
    tflite::tensor_utils::BatchQuantizeFloats(input_ptr, n_batch, n_input, quantized_input_ptr,
                                              input_sf, input_zp, asymmetric_quantize_inputs);
  }
  if (!is_aux_input_all_zeros)
  {
    tflite::tensor_utils::BatchQuantizeFloats(aux_input_ptr, n_batch, n_aux_input,
                                              quantized_aux_input_ptr, aux_input_sf, aux_input_zp,
                                              asymmetric_quantize_inputs);
  }
  if (!is_output_state_all_zeros)
  {
    tflite::tensor_utils::BatchQuantizeFloats(output_state_ptr, n_batch, n_output,
                                              quantized_output_state_ptr, output_state_sf,
                                              output_state_zp, asymmetric_quantize_inputs);
  }
  if (!use_cifg)
  {
    // Calculate the input gate. (If not CIFG.)
    CalculateLstmGateHybrid(
      quantized_input_ptr, input_sf, input_zp, input_to_input_weights_ptr,
      input_to_input_weights_ledger_ptr, input_to_input_weights_scale, input_to_input_row_sums,
      quantized_aux_input_ptr, aux_input_sf, aux_input_zp, aux_input_to_input_weights_ptr,
      aux_input_to_input_weights_scale, aux_input_to_input_row_sums, quantized_output_state_ptr,
      output_state_sf, output_state_zp, recurrent_to_input_weights_ptr,
      recurrent_to_input_weights_ledger_ptr, recurrent_to_input_weights_scale,
      recurrent_to_input_row_sums, cell_state_ptr, cell_to_input_weights_ptr,
      cell_to_input_weights_scale, input_layer_norm_coefficients_ptr, input_gate_bias_ptr, n_batch,
      n_input, n_aux_input, n_output, n_cell, kTfLiteActSigmoid, input_gate_scratch,
      is_input_all_zeros, is_aux_input_all_zeros, is_output_state_all_zeros, compute_row_sums,
      scaling_factors_scratch, recovered_cell_weights, accum_scratch_ptr);
  }
  // Calculate the forget gate.
  CalculateLstmGateHybrid(
    quantized_input_ptr, input_sf, input_zp, input_to_forget_weights_ptr,
    input_to_forget_weights_ledger_ptr, input_to_forget_weights_scale, input_to_forget_row_sums,
    quantized_aux_input_ptr, aux_input_sf, aux_input_zp, aux_input_to_forget_weights_ptr,
    aux_input_to_forget_weights_scale, aux_input_to_forget_row_sums, quantized_output_state_ptr,
    output_state_sf, output_state_zp, recurrent_to_forget_weights_ptr,
    recurrent_to_forget_weights_ledger_ptr, recurrent_to_forget_weights_scale,
    recurrent_to_forget_row_sums, cell_state_ptr, cell_to_forget_weights_ptr,
    cell_to_forget_weights_scale, forget_layer_norm_coefficients_ptr, forget_gate_bias_ptr, n_batch,
    n_input, n_aux_input, n_output, n_cell, kTfLiteActSigmoid, forget_gate_scratch,
    is_input_all_zeros, is_aux_input_all_zeros, is_output_state_all_zeros, compute_row_sums,
    scaling_factors_scratch, recovered_cell_weights, accum_scratch_ptr);
  // Calculate the cell update gate.
  CalculateLstmGateHybrid(
    quantized_input_ptr, input_sf, input_zp, input_to_cell_weights_ptr,
    input_to_cell_weights_ledger_ptr, input_to_cell_weights_scale, input_to_cell_row_sums,
    quantized_aux_input_ptr, aux_input_sf, aux_input_zp, aux_input_to_cell_weights_ptr,
    aux_input_to_cell_weights_scale, aux_input_to_cell_row_sums, quantized_output_state_ptr,
    output_state_sf, output_state_zp, recurrent_to_cell_weights_ptr,
    recurrent_to_cell_weights_ledger_ptr, recurrent_to_cell_weights_scale,
    recurrent_to_cell_row_sums,
    /*cell_state=*/nullptr, /*cell_to_gate_weights=*/nullptr,
    /*cell_to_gate_weights_scale=*/0.0f, cell_layer_norm_coefficients_ptr, cell_gate_bias_ptr,
    n_batch, n_input, n_aux_input, n_output, n_cell, params->activation, cell_gate_scratch,
    is_input_all_zeros, is_aux_input_all_zeros, is_output_state_all_zeros, compute_row_sums,
    scaling_factors_scratch, recovered_cell_weights, accum_scratch_ptr);
  // Update the cell state.
  UpdateLstmCellFloat(n_batch, n_cell, cell_state_ptr, input_gate_scratch, forget_gate_scratch,
                      cell_gate_scratch, use_cifg, params->cell_clip);
  // Calculate the output gate.
  CalculateLstmGateHybrid(
    quantized_input_ptr, input_sf, input_zp, input_to_output_weights_ptr,
    input_to_output_weights_ledger_ptr, input_to_output_weights_scale, input_to_output_row_sums,
    quantized_aux_input_ptr, aux_input_sf, aux_input_zp, aux_input_to_output_weights_ptr,
    aux_input_to_output_weights_scale, aux_input_to_output_row_sums, quantized_output_state_ptr,
    output_state_sf, output_state_zp, recurrent_to_output_weights_ptr,
    recurrent_to_output_weights_ledger_ptr, recurrent_to_output_weights_scale,
    recurrent_to_output_row_sums, cell_state_ptr, cell_to_output_weights_ptr,
    cell_to_output_weights_scale, output_layer_norm_coefficients_ptr, output_gate_bias_ptr, n_batch,
    n_input, n_aux_input, n_output, n_cell, kTfLiteActSigmoid, output_gate_scratch,
    is_input_all_zeros, is_aux_input_all_zeros, is_output_state_all_zeros, compute_row_sums,
    scaling_factors_scratch, recovered_cell_weights, accum_scratch_ptr);
  // Update the output state.
  CalculateLstmOutputHybrid(
    n_batch, n_cell, n_output, cell_state_ptr, output_gate_scratch, params->activation,
    projection_weights_ptr, projection_weights_ledger_ptr, projection_weights_scale,
    projection_bias_ptr, params->proj_clip, output_state_ptr, asymmetric_quantize_inputs,
    projection_weights_row_sums, compute_row_sums, scratch2, quantized_output_scratch, input_sf,
    input_zp, accum_scratch_ptr);
  // Copy output state to the output. Note that the output's rows may not be
  // contiguous (output_batch_leading_dim != n_output).
  for (int b = 0; b < n_batch; b++)
  {
    std::copy_n(output_state_ptr + b * n_output, n_output,
                output_ptr + b * output_batch_leading_dim);
  }
}

} // namespace eval_lstm
} // namespace kernels
} // namespace luci_interpreter
