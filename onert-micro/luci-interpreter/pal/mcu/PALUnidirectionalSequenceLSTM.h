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

#ifndef LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H
#define LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H

#include "core/KernelParams.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "fixedpoint/fixedpoint.h"

namespace luci_interpreter_pal
{
namespace lstm
{
// Possible fused activation functions.
typedef enum
{
  kTfLiteActNone = 0,
  kTfLiteActRelu,
  kTfLiteActReluN1To1, // min(max(-1, x), 1)
  kTfLiteActRelu6,     // min(max(0, x), 6)
  kTfLiteActTanh,
  kTfLiteActSignBit,
  kTfLiteActSigmoid,
} TfLiteFusedActivation;

inline int32_t multiply_by_quantized_multiplier(int32_t x, int32_t quantized_multiplier, int shift)
{
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(
    SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier), right_shift);
}

void cwise_mul(const int16_t *input_1, const int16_t *input_2, int32_t multiplier, int32_t shift,
               int32_t n_batch, int32_t n_input, int32_t output_zp, int8_t *output)
{
  for (int batch = 0; batch < n_batch; ++batch)
  {
    for (int i = 0; i < n_input; ++i)
    {
      const int index = batch * n_input + i;
      const int16_t a = input_1[index];
      const int16_t b = input_2[index];
      int32_t value = static_cast<int32_t>(a) * static_cast<int32_t>(b);
      value = multiply_by_quantized_multiplier(value, multiplier, shift);
      value -= output_zp;
      value = std::min(std::max(static_cast<int32_t>(-128), value), static_cast<int32_t>(127));

      output[index] = static_cast<int8_t>(value);
    }
  }
}

void sub1_vector(const int16_t *vector, int v_size, int16_t *result)
{
  static const int16_t kOne = 32767;
  for (int v = 0; v < v_size; v++)
  {
    *result++ = kOne - *vector++;
  }
}

template <typename T> void cwise_clipping(T *vector, const int v_size, const T &clipping_value)
{
  for (int i = 0; i < v_size; i++)
  {
    vector[i] = std::max(std::min(clipping_value, vector[i]), static_cast<T>(-clipping_value));
  }
}

void cwise_add(const int16_t *input_1, const int16_t *input_2, int n_batch, int n_input,
               int16_t *output)
{
  const int32_t kInt16Max = std::numeric_limits<int16_t>::max();
  const int32_t kInt16Min = std::numeric_limits<int16_t>::min();
  for (int batch = 0; batch < n_batch; ++batch)
  {
    for (int i = 0; i < n_input; ++i)
    {
      const int index = batch * n_input + i;
      int32_t sum = input_1[index] + input_2[index];
      const int32_t sum_clamped = std::min(kInt16Max, std::max(kInt16Min, sum));
      output[index] = static_cast<int16_t>(sum_clamped);
    }
  }
}

template <typename T> int count_leading_zeros(T integer_input)
{
  static_assert(std::is_unsigned<T>::value, "Only unsigned integer types handled.");
#if defined(__GNUC__)
  return integer_input ? __builtin_clz(integer_input) : std::numeric_limits<T>::digits;
#else
  if (integer_input == 0)
  {
    return std::numeric_limits<T>::digits;
  }

  const T one_in_leading_positive = static_cast<T>(1) << (std::numeric_limits<T>::digits - 1);
  int leading_zeros = 0;
  while (integer_input < one_in_leading_positive)
  {
    integer_input <<= 1;
    ++leading_zeros;
  }
  return leading_zeros;
#endif
}

inline void get_inv_sqrt_quantized_multiplier_exp(int32_t input, int reverse_shift,
                                                  int32_t *output_inv_sqrt, int *output_shift)
{
  if (input <= 1)
  {
    // Handle the input value 1 separately to avoid overflow in that case
    // in the general computation below (b/143972021). Also handle 0 as if it
    // were a 1. 0 is an invalid input here (divide by zero) and 1 is a valid
    // but rare/unrealistic input value. We can expect both to occur in some
    // incompletely trained models, but probably not in fully trained models.
    *output_inv_sqrt = std::numeric_limits<std::int32_t>::max();
    *output_shift = 0;
    return;
  }
  *output_shift = 11;
  while (input >= (1 << 29))
  {
    input /= 4;
    ++*output_shift;
  }
  const unsigned max_left_shift_bits = count_leading_zeros(static_cast<uint32_t>(input)) - 1;
  const unsigned max_left_shift_bit_pairs = max_left_shift_bits / 2;
  const unsigned left_shift_bit_pairs = max_left_shift_bit_pairs - 1;
  *output_shift -= left_shift_bit_pairs;
  input <<= 2 * left_shift_bit_pairs;
  using gemmlowp::FixedPoint;
  using gemmlowp::Rescale;
  using gemmlowp::SaturatingRoundingMultiplyByPOT;
  // Using 3 integer bits gives us enough room for the internal arithmetic in
  // this Newton-Raphson iteration.
  using F3 = FixedPoint<int32_t, 3>;
  using F0 = FixedPoint<int32_t, 0>;
  const F3 fixedpoint_input = F3::FromRaw(input >> 1);
  const F3 fixedpoint_half_input = SaturatingRoundingMultiplyByPOT<-1>(fixedpoint_input);
  const F3 fixedpoint_half_three =
    GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F3, (1 << 28) + (1 << 27), 1.5);
  // Newton-Raphson iteration
  // Naive unoptimized starting guess: x = 1
  F3 x = F3::One();
  // Naive unoptimized number of iterations: 5
  for (int i = 0; i < 5; i++)
  {
    const F3 x3 = Rescale<3>(x * x * x);
    x = Rescale<3>(fixedpoint_half_three * x - fixedpoint_half_input * x3);
  }
  const F0 fixedpoint_half_sqrt_2 =
    GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F0, 1518500250, std::sqrt(2.) / 2.);
  x = x * fixedpoint_half_sqrt_2;
  *output_inv_sqrt = x.raw();
  if (*output_shift < 0)
  {
    *output_inv_sqrt <<= -*output_shift;
    *output_shift = 0;
  }
  // Convert right shift (right is positive) to left shift.
  *output_shift *= reverse_shift;
}

void apply_layer_norm(const int16_t *input, const int16_t *layer_norm_weights, const int32_t *bias,
                      int32_t layer_norm_scale_a, int32_t layer_norm_scale_b,
                      int32_t variance_limit, int n_batch, int n_input, int16_t *output)
{
  // The square of std::pow(2, 10), which is the extra factor that makes sure
  // normalized values has enough resolution.
  static const int kTwoToPower20 = 1 << 20;
  const int32_t kInt16Max = std::numeric_limits<int16_t>::max();
  const int32_t kInt16Min = std::numeric_limits<int16_t>::min();
  for (int i = 0; i < n_batch; ++i)
  {
    int64_t sum = 0;
    int64_t sum_sq = 0;
    for (int j = 0; j < n_input; ++j)
    {
      const int32_t index = i * n_input + j;
      int32_t val = static_cast<int32_t>(input[index]);
      sum += val;
      sum_sq += val * val;
    }
    int32_t mean = static_cast<int32_t>(static_cast<int64_t>(sum) * 1024 / n_input);
    // TODO Avoids overflow but only works for POT n_input.
    int32_t temp = kTwoToPower20 / n_input;
    int64_t variance = sum_sq * temp - static_cast<int64_t>(mean) * static_cast<int64_t>(mean);
    int32_t variance2 = static_cast<int32_t>(variance / kTwoToPower20);
    if (variance2 < 1)
    {
      variance2 = variance_limit;
    }
    int32_t stddev_inverse_a;
    int stddev_inverse_b;
    get_inv_sqrt_quantized_multiplier_exp(variance2, /*reverse_shift*/ -1, &stddev_inverse_a,
                                          &stddev_inverse_b);

    for (int j = 0; j < n_input; ++j)
    {
      const int32_t index = i * n_input + j;
      int32_t val = static_cast<int32_t>(input[index]);
      int32_t shifted = 1024 * val - mean;
      int32_t rescaled =
        multiply_by_quantized_multiplier(shifted, stddev_inverse_a, stddev_inverse_b);
      // TODO: Saturate this.
      int64_t val3 = rescaled * layer_norm_weights[j] + bias[j];
      int32_t val4 = static_cast<int32_t>((val3 > 0 ? val3 + 512 : val3 - 512) / 1024);
      int32_t val5 =
        multiply_by_quantized_multiplier(val4, layer_norm_scale_a, layer_norm_scale_b + 12);
      val5 = std::min(std::max(kInt16Min, val5), kInt16Max);
      output[index] = static_cast<int16_t>(val5);
    }
  }
}

void vector_batch_vector_cwise_product_accumulate(const int16_t *vector, int v_size,
                                                  const int16_t *batch_vector, int n_batch,
                                                  int32_t multiplier, int shift, int16_t *result)
{
  for (int b = 0; b < n_batch; b++)
  {
    for (int v = 0; v < v_size; v++)
    {
      int32_t prod = vector[v] * *batch_vector++;
      prod = multiply_by_quantized_multiplier(prod, multiplier, shift);
      int32_t output = prod + *result;
      output =
        std::max(std::min(static_cast<int32_t>(32767), output), static_cast<int32_t>(-32768));
      *result++ = output;
    }
  }
}

template <typename T>
void matrix_batch_vector_multiply_accumulate(const int8_t *input, const int32_t *bias,
                                             const int8_t *input_to_gate_weights,
                                             int32_t multiplier, int32_t shift, int32_t n_batch,
                                             int32_t n_input, int32_t n_output, int32_t output_zp,
                                             T *output)
{
  const int16_t output_max = std::numeric_limits<T>::max();
  const int16_t output_min = std::numeric_limits<T>::min();
  for (int batch = 0; batch < n_batch; ++batch)
  {
    for (int row = 0; row < n_output; ++row)
    {
      int32_t acc = bias[row];
      for (int col = 0; col < n_input; ++col)
      {
        int8_t input_val = input[batch * n_input + col];
        int8_t weights_val = input_to_gate_weights[row * n_input + col];
        acc += input_val * weights_val;
      }
      acc = multiply_by_quantized_multiplier(acc, multiplier, shift);
      acc += output_zp;
      acc += output[batch * n_output + row];
      if (acc > output_max)
      {
        acc = output_max;
      }
      if (acc < output_min)
      {
        acc = output_min;
      }
      output[batch * n_output + row] = static_cast<T>(acc);
    }
  }
}

template <int IntegerBits>
void apply_tanh_impl(const int16_t *input, int32_t n_batch, int32_t n_input, int16_t *output)
{
  using FX = gemmlowp::FixedPoint<std::int16_t, IntegerBits>;
  using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
  for (int batch = 0; batch < n_batch; ++batch)
  {
    for (int i = 0; i < n_input; ++i)
    {
      const int index = batch * n_input + i;
      FX tanh_input = FX::FromRaw(input[index]);
      F0 tanh_output = gemmlowp::tanh(tanh_input);
      output[index] = tanh_output.raw();
    }
  }
}

void apply_tanh(int32_t integer_bits, const int16_t *input, int32_t n_batch, int32_t n_input,
                int16_t *output)
{
  assert(integer_bits <= 6);
#define DISPATCH_TANH(i)                                 \
  case i:                                                \
    apply_tanh_impl<i>(input, n_batch, n_input, output); \
    break;
  switch (integer_bits)
  {
    DISPATCH_TANH(0);
    DISPATCH_TANH(1);
    DISPATCH_TANH(2);
    DISPATCH_TANH(3);
    DISPATCH_TANH(4);
    DISPATCH_TANH(5);
    DISPATCH_TANH(6);
    default:
      return;
  }
#undef DISPATCH_TANH
}

// Calculates the output state tensor of an LSTM step. See Float and hybrid
// versions as well.
//
// Parameters:
//  - n_batch: batches: the number of distinct vectors in each array.
//  - n_cell, n_output: sizes of vectors.
//  - cell_state, output_gate: input vectors, size n_batch*n_cell.
//  - cell_state_scale: scaling of cell_state.
//  - hidden_scale_[a|b]: effective scale of cell_state.*output_gate
//  - hidden_zp: zero_point for cell_state.*output_gate
//  - projection_weights, proj_scale_[a|b], projection_bias:
//      constant inputs, describing projection matrix and bias.
//  - output_state_zp: zero point of output_state. (Input, calibrated value.)
//  - quantized_proj_clip: if > 0, clip the output of the projection.
//  - output_state: output vector, size n_batch*n_output. Must be contigous.
//  - scratch0: scratch area of size n_batch*n_cell
//  - scratch1: scratch area of size n_batch*n_cell
//  - scratch2: scratch area used by MatrixBatchVectorMultiplyAccumulate
void calculate_lstm_output_integer8x8_16(int n_batch, int n_cell, int n_output, int16_t *cell_state,
                                         int32_t cell_state_scale, const int16_t *output_gate,
                                         int32_t hidden_scale_a, int32_t hidden_scale_b,
                                         int32_t hidden_zp, const int8_t *projection_weights,
                                         int32_t proj_scale_a, int32_t proj_scale_b,
                                         const int32_t *projection_bias, int32_t output_state_zp,
                                         int8_t quantized_proj_clip, int8_t *output_state,
                                         int16_t *scratch0, int8_t *scratch1, int32_t *scratch2)
{
  // Note: unlike float/hybrid, the activation is always Tanh.
  apply_tanh(15 + cell_state_scale, cell_state, n_batch, n_cell, scratch0);
  const bool use_projection = (projection_weights != nullptr);

  if (use_projection)
  {
    // b/246629213 the projection operation assumes -hidden_zp in CwiseMul
    cwise_mul(output_gate, scratch0, hidden_scale_a, hidden_scale_b, n_batch, n_cell, -hidden_zp,
              scratch1);
    // Note: no bias like in float/hybrid
    std::fill_n(output_state, n_batch * n_output, 0);
    matrix_batch_vector_multiply_accumulate(scratch1, projection_bias, projection_weights,
                                            proj_scale_a, proj_scale_b, n_batch, n_cell, n_output,
                                            output_state_zp, output_state);
    if (quantized_proj_clip > 0)
    {
      cwise_clipping(output_state, n_batch * n_output, quantized_proj_clip);
    }
  }
  else
  {
    cwise_mul(output_gate, scratch0, hidden_scale_a, hidden_scale_b, n_batch, n_cell, hidden_zp,
              output_state);
  }
}

void cwise_mul(const int16_t *input_1, const int16_t *input_2, int n_batch, int n_input, int shift,
               int16_t *output)
{
  for (int batch = 0; batch < n_batch; ++batch)
  {
    for (int i = 0; i < n_input; ++i)
    {
      const int index = batch * n_input + i;
      const int16_t a = input_1[index];
      const int16_t b = input_2[index];
      const int32_t value = static_cast<int32_t>(a) * static_cast<int32_t>(b);
      output[index] = static_cast<int16_t>(gemmlowp::RoundingDivideByPOT(value, shift));
    }
  }
}

// Updates the LSTM cell state, used by both integer LSTM versions.
// Also see UpdateLstmCellFloat.
//
// Parameters:
//  - n_batch, n_cell: sizes of vectors
//  - cell_state: input/output vector, size n_batch*n_cell
//  - cell_state_scale: scaling factor of cell state.
//  - input_gate: input vector, size n_batch*n_cell.
//  - forget_gate: input/scratch vector, size n_batch*n_cell, always modified.
//  - cell_gate: input vector, size n_batch*n_cell.
//  - use_cifg: use 1-forget_gate instead of input_gate.
//  - clip: if > 0, clip the resulting cell state to [-clip, +clip].
void update_lstm_cell_integer(int n_batch, int n_cell, int16_t *cell_state,
                              int32_t cell_state_scale, const int16_t *input_gate,
                              int16_t *forget_gate, const int16_t *cell_gate, bool use_cifg,
                              int16_t clip)
{
  // Use the forget_gate array as scratch, as input_gate array is not allocated
  // in CIFG case. (Be careful not to write to the scratch before reading the
  // forget gate data.)
  int16_t *scratch = forget_gate;

  cwise_mul(forget_gate, cell_state, n_batch, n_cell, 15, cell_state);
  if (use_cifg)
  {
    sub1_vector(forget_gate, n_batch * n_cell, scratch);
    cwise_mul(scratch, cell_gate, n_batch, n_cell, 30 + cell_state_scale, scratch);
  }
  else
  {
    cwise_mul(input_gate, cell_gate, n_batch, n_cell, 30 + cell_state_scale, scratch);
  }
  cwise_add(cell_state, scratch, n_batch, n_cell, cell_state);

  if (clip > 0)
  {
    cwise_clipping(cell_state, n_batch * n_cell, clip);
  }
}

// Calculates a single LSTM gate, int8x8_16 version.
// Implements the same functionality as CalculateLstmGateFloat.
void calculate_lstm_gate_integer_8x8_16(
  // Input and weights
  const int8_t *input, const int8_t *input_to_gate_weights, const int32_t *input_to_gate_bias,
  const int32_t input_to_gate_scale_a, const int32_t input_to_gate_scale_b,
  // Output state and weights
  const int8_t *output_state, const int8_t *recurrent_to_gate_weights,
  const int32_t *recurrent_to_gate_bias, const int32_t recurrent_to_gate_scale_a,
  const int32_t recurrent_to_gate_scale_b,
  // Cell state and weights
  const int16_t *cell_state, const int16_t *cell_to_gate_weights,
  const int32_t cell_to_gate_scale_a, const int32_t cell_to_gate_scale_b,
  // Layer normalization parameters (layer norm LSTM)
  const int16_t *layer_norm_coefficients, const int32_t *layer_norm_bias,
  const int32_t layer_norm_input_scale_a, const int32_t layer_norm_input_scale_b,
  const int32_t layer_norm_variance_guard,
  // Array sizes
  const int n_batch, const int n_input, const int n_output, const int n_cell,
  const TfLiteFusedActivation activation,
  // Output
  int16_t *gate,
  // Parameters for performance optimizations
  // Scratch arrays
  int32_t *scratch5)
{
  const bool use_peephole = (cell_to_gate_weights != nullptr);
  const bool use_layer_norm = (layer_norm_coefficients != nullptr);

  // Initialize scratch buffers with zeros. Note that unlike float and hybrid
  // versions, bias is only used in layer normalization.
  memset(gate, 0, n_batch * n_cell * sizeof(int16_t));
  // For each batch and cell: compute input_weight * input.
  matrix_batch_vector_multiply_accumulate(input, input_to_gate_bias, input_to_gate_weights,
                                          input_to_gate_scale_a, input_to_gate_scale_b, n_batch,
                                          n_input, n_cell, 0, gate);
  // Note: no aux_input.
  // For each batch and cell: compute recurrent_weight * output_state.
  matrix_batch_vector_multiply_accumulate(
    output_state, recurrent_to_gate_bias, recurrent_to_gate_weights, recurrent_to_gate_scale_a,
    recurrent_to_gate_scale_b, n_batch, n_output, n_cell, 0, gate);
  // For each batch and cell: compute cell_weight * cell_state (peephole LSTM)
  if (use_peephole)
  {
    vector_batch_vector_cwise_product_accumulate(cell_to_gate_weights, n_output, cell_state,
                                                 n_batch, cell_to_gate_scale_a,
                                                 cell_to_gate_scale_b, gate);
  }
  // Do layer normalization (if layer norm LSTM)
  if (use_layer_norm)
  {
    apply_layer_norm(gate, layer_norm_coefficients, layer_norm_bias, layer_norm_input_scale_a,
                     layer_norm_input_scale_b, layer_norm_variance_guard, n_batch, n_cell, gate);
  }

  // Apply activation
  switch (activation)
  {
    case kTfLiteActSigmoid:
      tflite::reference_integer_ops::Logistic(
        0 /*data->input_multiplier*/, 0 /*data->input_left_shift */,
        n_batch * n_cell /*NumElements(input->dims)*/,
        gate /* tflite::micro::GetTensorData<int16_t>(input) */,
        gate /*tflite::micro::GetTensorData<int16_t>(output) */);

      break;
    case kTfLiteActTanh:
    {
      int32_t dims_data = n_batch * n_cell;
      tflite::RuntimeShape tanh_inp_shape = tflite::RuntimeShape(1, &dims_data);
      tflite::reference_integer_ops::Tanh(0, 0, tanh_inp_shape, gate, tanh_inp_shape, gate);
    }
    break;
    default:
      // Only Sigmoid or Tanh is used.
      TFLITE_ASSERT_FALSE;
  }
}

// Fully quantized lstm kernel for 16 bit gate matmul output.
//
// Input tensor of size n_batch * n_input:
//   input_ptr
//
// LSTM weights:
// Quantized input weights of size 'n_cell * n_input':
//   input_to_input_weight_ptr            - optional
//   input_to_forget_weight_ptr           - optional
//   input_to_cell_weight_ptr             - optional
//   input_to_output_weight_ptr           - optional
//
// Quantized recurrent weights of size 'n_cell * n_output':
//   recurrent_to_input_weight_ptr        - optional
//   recurrent_to_forget_weights_ptr
//   recurrent_to_cell_weights_ptr
//   recurrent_to_input_weights_ptr
//
// Quantized peephole weights of size 'n_cell', representing diagonal matrices.
//   cell_to_input_weights               - optional
//   cell_to_cell_weights                - optional
//   cell_to_output_weights              - optional
//
// Quantized projection weights of size 'n_output * n_cell'
//   projection_weight_ptr                     - optional
//
// Weight scales (scalars) for each of the weights above.
//   effective_input_to_input_scale_a    - optional
//   effective_input_to_input_scale_b    - optional
//   effective_input_to_forget_scale_a
//   effective_input_to_forget_scale_b
//   effective_input_to_cell_scale_a
//   effective_input_to_cell_scale_b
//   effective_input_to_output_scale_a
//   effective_input_to_output_scale_b
//   effective_recurrent_to_input_scale_a    - optional
//   effective_recurrent_to_input_scale_b    - optional
//   effective_recurrent_to_forget_scale_a
//   effective_recurrent_to_forget_scale_b
//   effective_recurrent_to_cell_scale_a
//   effective_recurrent_to_cell_scale_b
//   effective_recurrent_to_output_scale_a
//   effective_recurrent_to_output_scale_b
//   effective_proj_scale_a                  - optional
//   effective_proj_scale_b                  - optional
//
// Gate biases of size 'n_cell':
//   input_gate_bias_ptr                 - optional
//   forget_gate_bias_ptr
//   cell_gate_bias_ptr
//   output_gate_bias_ptr
//
// Layer norm coefficients of size 'n_cell', representing diagonal matrices.
//   layer_norm_input_weight_ptr    - optional
//   layer_norm_forget_weight_ptr   - optional
//   layer_norm_cell_weight_ptr     - optional
//   layer_norm_output_weight_ptr   - optional
//
// Layer norm scales of size 'n_cell'.
//   layer_norm_input_scale_a     - optional
//   layer_norm_input_scale_b     - optional
//   layer_norm_forget_scale_a    - optional
//   layer_norm_forget_scale_b    - optional
//   layer_norm_cell_scale_a      - optional
//   layer_norm_cell_scale_b      - optional
//   layer_norm_output_scale_a    - optional
//   layer_norm_output_scale_b    - optional
//
// Scalar values:
//   quantized_cell_clip: quantized clip value for cell.
//   quantized_proj_clip: quantized clip value for projection.
//   cell_state_scale: the power of two scale for cell state.
//
// Zero points:
//   output_state_zp: zero point of output state
//   hidden_zp: zero point for hidden state.
//
// Temporary pre-allocated storage for the calculation. Each is of size n_cell *
// n_batch.
//   scratch0
//   scratch1
//   scratch2
//   scratch3
//   scratch4
//   scratch5: this scratch buffer is created purely for optimizing the
//              MatrixBatchVectorMultiplyAccumulate.
//
// Outputs:
//   output_state_ptr - size 'n_batch * n_output'
//   cell_state_ptr   - size 'n_batch * n_cell'
//   output_ptr       - size 'n_batch * n_output'
// TODO(b/159947023): scratch0 is not used if (!cifg). Don't allocate then.
void lstm_step_integer_8x8_16(
  const int8_t *input_ptr, const int8_t *input_to_input_weight_ptr,
  int32_t effective_input_to_input_scale_a, int32_t effective_input_to_input_scale_b,
  const int8_t *input_to_forget_weight_ptr, int32_t effective_input_to_forget_scale_a,
  int32_t effective_input_to_forget_scale_b, const int8_t *input_to_cell_weight_ptr,
  int32_t effective_input_to_cell_scale_a, int32_t effective_input_to_cell_scale_b,
  const int8_t *input_to_output_weight_ptr, int32_t effective_input_to_output_scale_a,
  int32_t effective_input_to_output_scale_b, const int8_t *recurrent_to_input_weight_ptr,
  int32_t effective_recurrent_to_input_scale_a, int32_t effective_recurrent_to_input_scale_b,
  const int8_t *recurrent_to_forget_weight_ptr, int32_t effective_recurrent_to_forget_scale_a,
  int32_t effective_recurrent_to_forget_scale_b, const int8_t *recurrent_to_cell_weight_ptr,
  int32_t effective_recurrent_to_cell_scale_a, int32_t effective_recurrent_to_cell_scale_b,
  const int8_t *recurrent_to_output_weight_ptr, int32_t effective_recurrent_to_output_scale_a,
  int32_t effective_recurrent_to_output_scale_b, const int16_t *cell_to_input_weight_ptr,
  int32_t effective_cell_to_input_scale_a, int32_t effective_cell_to_input_scale_b,
  const int16_t *cell_to_forget_weight_ptr, int32_t effective_cell_to_forget_scale_a,
  int32_t effective_cell_to_forget_scale_b, const int16_t *cell_to_output_weight_ptr,
  int32_t effective_cell_to_output_scale_a, int32_t effective_cell_to_output_scale_b,
  const int8_t *projection_weight_ptr, int32_t effective_proj_scale_a,
  int32_t effective_proj_scale_b, int32_t hidden_zp, int32_t effective_hidden_scale_a,
  int32_t effective_hidden_scale_b, const int16_t *layer_norm_input_weight_ptr,
  int32_t layer_norm_input_scale_a, int32_t layer_norm_input_scale_b,
  const int16_t *layer_norm_forget_weight_ptr, int32_t layer_norm_forget_scale_a,
  int32_t layer_norm_forget_scale_b, const int16_t *layer_norm_cell_weight_ptr,
  int32_t layer_norm_cell_scale_a, int32_t layer_norm_cell_scale_b,
  const int16_t *layer_norm_output_weight_ptr, int32_t layer_norm_output_scale_a,
  int32_t layer_norm_output_scale_b, const int32_t *input_gate_bias_ptr,
  const int32_t *forget_gate_bias_ptr, const int32_t *cell_gate_bias_ptr,
  const int32_t *output_gate_bias_ptr, int16_t quantized_cell_clip, int8_t quantized_proj_clip,
  int32_t cell_state_scale, int32_t input_variance_guard, int32_t forget_variance_guard,
  int32_t cell_variance_guard, int32_t output_variance_guard,
  const int32_t *input_to_forget_effective_bias, const int32_t *recurrent_to_forget_effective_bias,
  const int32_t *input_to_cell_effective_bias, const int32_t *recurrent_to_cell_effective_bias,
  const int32_t *input_to_output_effective_bias, const int32_t *recurrent_to_output_effective_bias,
  const int32_t *input_to_input_effective_bias, const int32_t *recurrent_to_input_effective_bias,
  const int32_t *projection_effective_bias, int n_batch, int n_cell, int n_input, int n_output,
  int8_t *output_state_ptr, int32_t output_state_zp, int16_t *cell_state_ptr, int8_t *output_ptr,
  int16_t *scratch0, int16_t *scratch1, int16_t *scratch2, int16_t *scratch3, int8_t *scratch4,
  int32_t *scratch5)
{
  // Make named scratch buffers for the different gates.
  int16_t *input_gate_scratch = scratch0;
  int16_t *forget_gate_scratch = scratch1;
  int16_t *cell_gate_scratch = scratch2;
  int16_t *output_gate_scratch = scratch3;

  // Since we have already checked that weights are all there or none, we
  // can check the existence of only one to the get the condition.
  const bool use_cifg = (input_to_input_weight_ptr == nullptr);

  LUCI_INTERPRETER_CHECK(input_to_forget_effective_bias);
  LUCI_INTERPRETER_CHECK(recurrent_to_forget_effective_bias);
  LUCI_INTERPRETER_CHECK(input_to_cell_effective_bias);
  LUCI_INTERPRETER_CHECK(recurrent_to_cell_effective_bias);
  LUCI_INTERPRETER_CHECK(input_to_output_effective_bias);
  LUCI_INTERPRETER_CHECK(recurrent_to_output_effective_bias);

  if (!use_cifg)
  {
    LUCI_INTERPRETER_CHECK(input_to_input_effective_bias);
    LUCI_INTERPRETER_CHECK(recurrent_to_input_effective_bias);
  }

  const bool use_projection = (projection_weight_ptr != nullptr);
  if (use_projection)
  {
    LUCI_INTERPRETER_CHECK(projection_effective_bias);
  }

  if (!use_cifg)
  {
    // Calculate the input gate. (If not CIFG.)
    calculate_lstm_gate_integer_8x8_16(
      input_ptr, input_to_input_weight_ptr, input_to_input_effective_bias,
      effective_input_to_input_scale_a, effective_input_to_input_scale_b, output_state_ptr,
      recurrent_to_input_weight_ptr, recurrent_to_input_effective_bias,
      effective_recurrent_to_input_scale_a, effective_recurrent_to_input_scale_b, cell_state_ptr,
      cell_to_input_weight_ptr, effective_cell_to_input_scale_a, effective_cell_to_input_scale_b,
      layer_norm_input_weight_ptr, input_gate_bias_ptr, layer_norm_input_scale_a,
      layer_norm_input_scale_b, input_variance_guard, n_batch, n_input, n_output, n_cell,
      kTfLiteActSigmoid, input_gate_scratch, scratch5);
  }

  // Calculate the forget gate.
  calculate_lstm_gate_integer_8x8_16(
    input_ptr, input_to_forget_weight_ptr, input_to_forget_effective_bias,
    effective_input_to_forget_scale_a, effective_input_to_forget_scale_b, output_state_ptr,
    recurrent_to_forget_weight_ptr, recurrent_to_forget_effective_bias,
    effective_recurrent_to_forget_scale_a, effective_recurrent_to_forget_scale_b, cell_state_ptr,
    cell_to_forget_weight_ptr, effective_cell_to_forget_scale_a, effective_cell_to_forget_scale_b,
    layer_norm_forget_weight_ptr, forget_gate_bias_ptr, layer_norm_forget_scale_a,
    layer_norm_forget_scale_b, forget_variance_guard, n_batch, n_input, n_output, n_cell,
    kTfLiteActSigmoid, forget_gate_scratch, scratch5);

  // Calculate the cell update gate.
  calculate_lstm_gate_integer_8x8_16(
    input_ptr, input_to_cell_weight_ptr, input_to_cell_effective_bias,
    effective_input_to_cell_scale_a, effective_input_to_cell_scale_b, output_state_ptr,
    recurrent_to_cell_weight_ptr, recurrent_to_cell_effective_bias,
    effective_recurrent_to_cell_scale_a, effective_recurrent_to_cell_scale_b, cell_state_ptr,
    /*cell_to_gate_weights=*/nullptr, /*cell_to_gate_scale_a=*/0,
    /*cell_to_gate_scale_b=*/0, layer_norm_cell_weight_ptr, cell_gate_bias_ptr,
    layer_norm_cell_scale_a, layer_norm_cell_scale_b, cell_variance_guard, n_batch, n_input,
    n_output, n_cell, kTfLiteActTanh, cell_gate_scratch, scratch5);

  // Update the cell state.
  update_lstm_cell_integer(n_batch, n_cell, cell_state_ptr, cell_state_scale, input_gate_scratch,
                           forget_gate_scratch, cell_gate_scratch, use_cifg, quantized_cell_clip);

  // Calculate the output gate.
  calculate_lstm_gate_integer_8x8_16(
    input_ptr, input_to_output_weight_ptr, input_to_output_effective_bias,
    effective_input_to_output_scale_a, effective_input_to_output_scale_b, output_state_ptr,
    recurrent_to_output_weight_ptr, recurrent_to_output_effective_bias,
    effective_recurrent_to_output_scale_a, effective_recurrent_to_output_scale_b, cell_state_ptr,
    cell_to_output_weight_ptr, effective_cell_to_output_scale_a, effective_cell_to_output_scale_b,
    layer_norm_output_weight_ptr, output_gate_bias_ptr, layer_norm_output_scale_a,
    layer_norm_output_scale_b, output_variance_guard, n_batch, n_input, n_output, n_cell,
    kTfLiteActSigmoid, output_gate_scratch, scratch5);

  // Update the output state.
  calculate_lstm_output_integer8x8_16(
    n_batch, n_cell, n_output, cell_state_ptr, cell_state_scale, output_gate_scratch,
    effective_hidden_scale_a, effective_hidden_scale_b, hidden_zp, projection_weight_ptr,
    effective_proj_scale_a, effective_proj_scale_b, projection_effective_bias, output_state_zp,
    quantized_proj_clip, output_state_ptr, scratch0, scratch4, scratch5);
  // Copy output state to the output. Note that unlike float or hybrid, output
  // is always contiguous.
  std::copy_n(output_state_ptr, n_batch * n_output, output_ptr);
}

} // namespace lstm

void eval_integer_8x8_16_lstm(
  const luci_interpreter::Tensor *input, const luci_interpreter::Tensor *input_to_input_weights,
  const luci_interpreter::Tensor *input_to_forget_weights,
  const luci_interpreter::Tensor *input_to_cell_weights,
  const luci_interpreter::Tensor *input_to_output_weights,
  const luci_interpreter::Tensor *recurrent_to_input_weights,
  const luci_interpreter::Tensor *recurrent_to_forget_weights,
  const luci_interpreter::Tensor *recurrent_to_cell_weights,
  const luci_interpreter::Tensor *recurrent_to_output_weights,
  const luci_interpreter::Tensor *cell_to_input_weights,
  const luci_interpreter::Tensor *cell_to_forget_weights,
  const luci_interpreter::Tensor *cell_to_output_weights,
  const luci_interpreter::Tensor *input_layer_norm_coefficients,
  const luci_interpreter::Tensor *forget_layer_norm_coefficients,
  const luci_interpreter::Tensor *cell_layer_norm_coefficients,
  const luci_interpreter::Tensor *output_layer_norm_coefficients,
  const luci_interpreter::Tensor *input_gate_bias, const luci_interpreter::Tensor *forget_gate_bias,
  const luci_interpreter::Tensor *cell_gate_bias, const luci_interpreter::Tensor *output_gate_bias,
  const luci_interpreter::Tensor *projection_weights,
  const luci_interpreter::Tensor *projection_bias,
  const luci_interpreter::UnidirectionalSequenceLSTMParams &params, bool forward_sequence,
  bool time_major, const luci_interpreter::IntegerLSTMParams &integer_lstm_param,
  int32_t output_state_zp, luci_interpreter::Tensor *output_state,
  luci_interpreter::Tensor *cell_state, luci_interpreter::Tensor *output, int16_t *scratch0,
  int16_t *scratch1, int16_t *scratch2, int16_t *scratch3, int8_t *scratch4, int32_t *scratch5)
{
  const auto input_shape = input->shape();
  LUCI_INTERPRETER_CHECK(input_shape.num_dims() >= 2 && input_shape.num_dims() <= 3);

  const int n_input = input_shape.dim(input_shape.num_dims() - 1);
  int max_time, n_batch;
  if (input_shape.num_dims() == 2)
  {
    max_time = 1;
    n_batch = input_shape.dim(0);
  }
  else
  {
    max_time = (time_major) ? input_shape.dim(0) : input_shape.dim(1);
    n_batch = (time_major) ? input_shape.dim(1) : input_shape.dim(0);
  }

  // n_cell and n_output will be the same size when there is no projection.
  const int n_cell = input_to_output_weights->shape().dim(0);
  const int n_output = recurrent_to_output_weights->shape().dim(1);

  // Get params for time/batch/sequence.
  const int output_batch_leading_dim = output->shape().dim(output->shape().num_dims() - 1);

  if (time_major)
  {
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;

    for (int t = 0; t < max_time; t++)
    {
      const int t_rel = t;
      int8_t *output_ptr =
        luci_interpreter::kernels::getTensorData<int8_t>(output) + t_rel * output_step;
      const int8_t *input_ptr =
        luci_interpreter::kernels::getTensorData<int8_t>(input) + t_rel * input_step;

      lstm::lstm_step_integer_8x8_16(
        input_ptr,
        input_to_input_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int8_t>(input_to_input_weights),
        integer_lstm_param.effective_input_to_input_scale_a,
        integer_lstm_param.effective_input_to_input_scale_b,
        input_to_forget_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int8_t>(input_to_forget_weights),
        integer_lstm_param.effective_input_to_forget_scale_a,
        integer_lstm_param.effective_input_to_forget_scale_b,
        input_to_cell_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int8_t>(input_to_cell_weights),
        integer_lstm_param.effective_input_to_cell_scale_a,
        integer_lstm_param.effective_input_to_cell_scale_b,
        input_to_output_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int8_t>(input_to_output_weights),
        integer_lstm_param.effective_input_to_output_scale_a,
        integer_lstm_param.effective_input_to_output_scale_b,
        recurrent_to_input_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_input_weights),
        integer_lstm_param.effective_recurrent_to_input_scale_a,
        integer_lstm_param.effective_recurrent_to_input_scale_b,
        recurrent_to_forget_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_forget_weights),
        integer_lstm_param.effective_recurrent_to_forget_scale_a,
        integer_lstm_param.effective_recurrent_to_forget_scale_b,
        recurrent_to_cell_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_cell_weights),
        integer_lstm_param.effective_recurrent_to_cell_scale_a,
        integer_lstm_param.effective_recurrent_to_cell_scale_b,
        recurrent_to_output_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_output_weights),
        integer_lstm_param.effective_recurrent_to_output_scale_a,
        integer_lstm_param.effective_recurrent_to_output_scale_b,
        cell_to_input_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int16_t>(cell_to_input_weights),
        integer_lstm_param.effective_cell_to_input_scale_a,
        integer_lstm_param.effective_cell_to_input_scale_b,
        cell_to_forget_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int16_t>(cell_to_forget_weights),
        integer_lstm_param.effective_cell_to_forget_scale_a,
        integer_lstm_param.effective_cell_to_forget_scale_b,
        cell_to_output_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int16_t>(cell_to_output_weights),
        integer_lstm_param.effective_cell_to_output_scale_a,
        integer_lstm_param.effective_cell_to_output_scale_b,
        projection_weights == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int8_t>(projection_weights),
        integer_lstm_param.effective_proj_scale_a, integer_lstm_param.effective_proj_scale_b,
        integer_lstm_param.hidden_zp, integer_lstm_param.effective_hidden_scale_a,
        integer_lstm_param.effective_hidden_scale_b,
        input_layer_norm_coefficients == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int16_t>(input_layer_norm_coefficients),
        integer_lstm_param.layer_norm_input_scale_a, integer_lstm_param.layer_norm_input_scale_b,
        forget_layer_norm_coefficients == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int16_t>(forget_layer_norm_coefficients),
        integer_lstm_param.layer_norm_forget_scale_a, integer_lstm_param.layer_norm_forget_scale_b,
        cell_layer_norm_coefficients == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int16_t>(cell_layer_norm_coefficients),
        integer_lstm_param.layer_norm_cell_scale_a, integer_lstm_param.layer_norm_cell_scale_b,
        output_layer_norm_coefficients == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int16_t>(output_layer_norm_coefficients),
        integer_lstm_param.layer_norm_output_scale_a, integer_lstm_param.layer_norm_output_scale_b,
        input_gate_bias == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int32_t>(input_gate_bias),
        forget_gate_bias == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int32_t>(forget_gate_bias),
        cell_gate_bias == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int32_t>(cell_gate_bias),
        output_gate_bias == nullptr
          ? nullptr
          : luci_interpreter::kernels::getTensorData<int32_t>(output_gate_bias),
        integer_lstm_param.quantized_cell_clip, integer_lstm_param.quantized_proj_clip,
        integer_lstm_param.cell_scale, integer_lstm_param.input_variance_guard,
        integer_lstm_param.forget_variance_guard, integer_lstm_param.cell_variance_guard,
        integer_lstm_param.output_variance_guard,
        integer_lstm_param.input_to_forget_effective_bias.data(),
        integer_lstm_param.recurrent_to_forget_effective_bias.data(),
        integer_lstm_param.input_to_cell_effective_bias.data(),
        integer_lstm_param.recurrent_to_cell_effective_bias.data(),
        integer_lstm_param.input_to_output_effective_bias.data(),
        integer_lstm_param.recurrent_to_output_effective_bias.data(),
        integer_lstm_param.input_to_input_effective_bias.data(),
        integer_lstm_param.recurrent_to_input_effective_bias.data(),
        integer_lstm_param.projection_effective_bias.data(), n_batch, n_cell, n_input, n_output,
        luci_interpreter::kernels::getTensorData<int8_t>(output_state), output_state_zp,
        luci_interpreter::kernels::getTensorData<int16_t>(cell_state), output_ptr, scratch0,
        scratch1, scratch2, scratch3, scratch4, scratch5);
    }
  }
  else
  {
    for (int b = 0; b < n_batch; b++)
    {
      const int input_step = n_input;
      const int output_step = output_batch_leading_dim;

      for (int t = 0; t < max_time; t++)
      {
        // If this is the forward_sequence, step forward, otherwise step
        // backwards.
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        const int time_offset = b * max_time + t_rel;
        const int8_t *input_ptr =
          luci_interpreter::kernels::getTensorData<int8_t>(input) + time_offset * input_step;
        int8_t *output_ptr =
          luci_interpreter::kernels::getTensorData<int8_t>(output) + time_offset * output_step;

        // Offset the {output,cell}_state pointers to the right batch.
        int8_t *output_state_ptr = luci_interpreter::kernels::getTensorData<int8_t>(output_state) +
                                   b * output_batch_leading_dim;
        int16_t *cell_state_ptr =
          luci_interpreter::kernels::getTensorData<int16_t>(cell_state) + b * n_cell;

        lstm::lstm_step_integer_8x8_16(
          input_ptr,
          input_to_input_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int8_t>(input_to_input_weights),
          integer_lstm_param.effective_input_to_input_scale_a,
          integer_lstm_param.effective_input_to_input_scale_b,
          input_to_forget_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int8_t>(input_to_forget_weights),
          integer_lstm_param.effective_input_to_forget_scale_a,
          integer_lstm_param.effective_input_to_forget_scale_b,
          input_to_cell_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int8_t>(input_to_cell_weights),
          integer_lstm_param.effective_input_to_cell_scale_a,
          integer_lstm_param.effective_input_to_cell_scale_b,
          input_to_output_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int8_t>(input_to_output_weights),
          integer_lstm_param.effective_input_to_output_scale_a,
          integer_lstm_param.effective_input_to_output_scale_b,
          recurrent_to_input_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_input_weights),
          integer_lstm_param.effective_recurrent_to_input_scale_a,
          integer_lstm_param.effective_recurrent_to_input_scale_b,
          recurrent_to_forget_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_forget_weights),
          integer_lstm_param.effective_recurrent_to_forget_scale_a,
          integer_lstm_param.effective_recurrent_to_forget_scale_b,
          recurrent_to_cell_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_cell_weights),
          integer_lstm_param.effective_recurrent_to_cell_scale_a,
          integer_lstm_param.effective_recurrent_to_cell_scale_b,
          recurrent_to_output_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int8_t>(recurrent_to_output_weights),
          integer_lstm_param.effective_recurrent_to_output_scale_a,
          integer_lstm_param.effective_recurrent_to_output_scale_b,
          cell_to_input_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int16_t>(cell_to_input_weights),
          integer_lstm_param.effective_cell_to_input_scale_a,
          integer_lstm_param.effective_cell_to_input_scale_b,
          cell_to_forget_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int16_t>(cell_to_forget_weights),
          integer_lstm_param.effective_cell_to_forget_scale_a,
          integer_lstm_param.effective_cell_to_forget_scale_b,
          cell_to_output_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int16_t>(cell_to_output_weights),
          integer_lstm_param.effective_cell_to_output_scale_a,
          integer_lstm_param.effective_cell_to_output_scale_b,
          projection_weights == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int8_t>(projection_weights),
          integer_lstm_param.effective_proj_scale_a, integer_lstm_param.effective_proj_scale_b,
          integer_lstm_param.hidden_zp, integer_lstm_param.effective_hidden_scale_a,
          integer_lstm_param.effective_hidden_scale_b,
          input_layer_norm_coefficients == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int16_t>(input_layer_norm_coefficients),
          integer_lstm_param.layer_norm_input_scale_a, integer_lstm_param.layer_norm_input_scale_b,
          forget_layer_norm_coefficients == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int16_t>(forget_layer_norm_coefficients),
          integer_lstm_param.layer_norm_forget_scale_a,
          integer_lstm_param.layer_norm_forget_scale_b,
          cell_layer_norm_coefficients == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int16_t>(cell_layer_norm_coefficients),
          integer_lstm_param.layer_norm_cell_scale_a, integer_lstm_param.layer_norm_cell_scale_b,
          output_layer_norm_coefficients == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int16_t>(output_layer_norm_coefficients),
          integer_lstm_param.layer_norm_output_scale_a,
          integer_lstm_param.layer_norm_output_scale_b,
          input_gate_bias == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int32_t>(input_gate_bias),
          forget_gate_bias == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int32_t>(forget_gate_bias),
          cell_gate_bias == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int32_t>(cell_gate_bias),
          output_gate_bias == nullptr
            ? nullptr
            : luci_interpreter::kernels::getTensorData<int32_t>(output_gate_bias),
          integer_lstm_param.quantized_cell_clip, integer_lstm_param.quantized_proj_clip,
          integer_lstm_param.cell_scale, integer_lstm_param.input_variance_guard,
          integer_lstm_param.forget_variance_guard, integer_lstm_param.cell_variance_guard,
          integer_lstm_param.output_variance_guard,
          integer_lstm_param.input_to_forget_effective_bias.data(),
          integer_lstm_param.recurrent_to_forget_effective_bias.data(),
          integer_lstm_param.input_to_cell_effective_bias.data(),
          integer_lstm_param.recurrent_to_cell_effective_bias.data(),
          integer_lstm_param.input_to_output_effective_bias.data(),
          integer_lstm_param.recurrent_to_output_effective_bias.data(),
          integer_lstm_param.input_to_input_effective_bias.data(),
          integer_lstm_param.recurrent_to_input_effective_bias.data(),
          integer_lstm_param.projection_effective_bias.data(), /*n_batch=*/1, n_cell, n_input,
          n_output, output_state_ptr, output_state_zp, cell_state_ptr, output_ptr, scratch0,
          scratch1, scratch2, scratch3, scratch4, scratch5);
      }
    }
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_H
