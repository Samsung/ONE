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

#ifndef LUCI_INTERPRETER_PAL_SVDF_H
#define LUCI_INTERPRETER_PAL_SVDF_H

#include <tensorflow/lite/kernels/internal/reference/svdf.h>

namespace luci_interpreter_pal
{
static inline void
IntegerSVDF(const TfLiteSVDFParams &params, const tflite::RuntimeShape &input_shape,
            const int8_t *input_data, const tflite::RuntimeShape &weight_feature_shape,
            const int8_t *weight_feature_data, const tflite::RuntimeShape &weight_time_shape,
            const int16_t *weight_time_data, const tflite::RuntimeShape &bias_shape,
            const int32_t *bias_data, int16_t *activation_state_data,
            const tflite::RuntimeShape &output_shape, int8_t *output_data, int32_t *scratchpad_data,
            int32_t *output_temp_data, int32_t scale_1_a, int scale_1_b, int32_t scale_2_a,
            int scale_2_b, int32_t input_zp, int32_t output_zp)
{
  const int n_rank = params.rank;
  const int n_batch = input_shape.Dims(0);
  const int n_input = input_shape.Dims(1);
  const int n_filter = weight_feature_shape.Dims(0);
  const int n_unit = n_filter / n_rank;
  const int n_memory = weight_time_shape.Dims(1);

  // Left shift the activation_state.
  {
    int16_t *new_state_start = activation_state_data;
    const int16_t *old_state_start = activation_state_data + 1;
    const int16_t *old_state_end = activation_state_data + n_batch * n_filter * n_memory;
    while (old_state_start != old_state_end)
    {
      *new_state_start++ = *old_state_start++;
    }
  }

  // Note: no need to clear the latest activation, matmul is not accumulative.

  // Feature matmul.
  {
    const int32_t output_max = std::numeric_limits<int16_t>::max();
    const int32_t output_min = std::numeric_limits<int16_t>::min();
    int16_t *result_in_batch = activation_state_data + (n_memory - 1);
    for (int b = 0; b < n_batch; b++)
    {
      const int8_t *matrix_ptr = weight_feature_data;
      for (int r = 0; r < n_filter; r++)
      {
        int32_t dot_prod = 0;
        const int8_t *vector_in_batch = input_data + b * n_input;
        for (int c = 0; c < n_input; c++)
        {
          dot_prod += *matrix_ptr++ * (*vector_in_batch++ - input_zp);
        }
        dot_prod = tflite::MultiplyByQuantizedMultiplier(dot_prod, scale_1_a, scale_1_b);
        dot_prod = std::min(std::max(output_min, dot_prod), output_max);
        // This assumes state is symmetrically quantized. Otherwise last bit of
        // state should be initialized to its zero point and accumulate the
        // dot_prod.
        // Equivalent as the following:
        //     result_in_batch = zero point, which happens to be zero.
        //     result_in_batch += dot_prod_56.
        *result_in_batch = dot_prod;
        result_in_batch += n_memory;
      }
    }
  }

  // Time.
  {
    for (int b = 0; b < n_batch; ++b)
    {
      int32_t *scratch_ptr_batch = scratchpad_data + b * n_filter;

      // Perform batched vector dot product:
      const int16_t *vector1_ptr = weight_time_data;
      const int16_t *vector2_ptr = activation_state_data + b * n_memory * n_filter;

      for (int i = 0; i < n_filter; i++)
      {
        *scratch_ptr_batch = 0;
        for (int j = 0; j < n_memory; j++)
        {
          *scratch_ptr_batch += *vector1_ptr++ * *vector2_ptr++;
        }
        scratch_ptr_batch++;
      }
    }
  }

  // Reduce, add bias, rescale, activation.
  {
    // Add bias.
    if (bias_data)
    {
      // Vector batch assign:
      for (int i = 0; i < n_batch; ++i)
      {
        int32_t *output_ptr = output_temp_data + i * n_unit;
        const int32_t *bias_ptr = bias_data;
        for (int j = 0; j < n_unit; ++j)
        {
          *output_ptr++ = *bias_ptr++;
        }
      }
    }
    else
    {
      int32_t *output_ptr = output_temp_data;
      for (int i = 0; i < n_batch * n_unit; ++i)
      {
        *output_ptr++ = 0;
      }
    }

    // Reduce.
    for (int b = 0; b < n_batch; ++b)
    {
      int32_t *output_temp_ptr = output_temp_data + b * n_unit;
      int32_t *scratch_ptr_batch = scratchpad_data + b * n_filter;

      // Reduction sum vector
      for (int i = 0; i < n_unit; ++i)
      {
        for (int j = 0; j < n_rank; ++j)
        {
          output_temp_ptr[i] += *scratch_ptr_batch++;
        }
      }
    }

    // Rescale.
    const int32_t output_max = std::numeric_limits<int8_t>::max();
    const int32_t output_min = std::numeric_limits<int8_t>::min();
    for (int i = 0; i < n_batch * n_unit; ++i)
    {
      int32_t x1 = output_temp_data[i];
      int32_t x2 = tflite::MultiplyByQuantizedMultiplier(x1, scale_2_a, scale_2_b);
      int32_t x3 = x2 + output_zp;
      int32_t x4 = std::min(std::max(output_min, x3), output_max);
      output_data[i] = static_cast<int8_t>(x4);
    }
  }
}
static inline void
FloatSVDF(const TfLiteSVDFParams &params, const tflite::RuntimeShape &input_shape,
          const float *input_data, const tflite::RuntimeShape &weight_feature_shape,
          const float *weight_feature_data, const tflite::RuntimeShape &weight_time_shape,
          const float *weight_time_data, const tflite::RuntimeShape &bias_shape,
          const float *bias_data, float *scratchpad_data, float *activation_state_data,
          const tflite::RuntimeShape &output_shape, float *output_data)
{
  const int32_t rank = params.rank;
  const int32_t batch_size = input_shape.Dims(0);
  const int32_t input_size = input_shape.Dims(1);
  const int32_t num_filters = weight_feature_shape.Dims(0);
  const int32_t num_units = num_filters / rank;
  const int32_t memory_size = weight_time_shape.Dims(1);

  // Left shift the activation_state.
  {
    float *new_state_start = activation_state_data;
    const float *old_state_start = activation_state_data + 1;
    const float *old_state_end = activation_state_data + batch_size * num_filters * memory_size;
    while (old_state_start != old_state_end)
    {
      *new_state_start++ = *old_state_start++;
    }
  }

  // Note: no need to clear the latest activation, matmul is not accumulative.

  // Compute conv1d(inputs, weights_feature).
  // The activation_state's rightmost column is used to save current cycle
  // activation. This is achieved by starting at state_ptr[memory_size - 1] and
  // having the stride equal to memory_size.

  // Perform batched matrix vector multiply operation:
  {
    const float *matrix = weight_feature_data;
    const float *vector = input_data;
    float *result = &activation_state_data[memory_size - 1];
    float *result_in_batch = result;
    for (int i = 0; i < batch_size; ++i)
    {
      const float *matrix_ptr = matrix;
      for (int j = 0; j < num_filters; ++j)
      {
        float dot_prod = 0.0f;
        const float *vector_in_batch = vector + i * input_size;
        for (int k = 0; k < input_size; ++k)
        {
          dot_prod += *matrix_ptr++ * *vector_in_batch++;
        }
        *result_in_batch = dot_prod;
        result_in_batch += memory_size;
      }
    }
  }

  tflite::reference_ops::ApplyTimeWeightsBiasAndActivation(
    batch_size, memory_size, num_filters, num_units, rank, weight_time_data, bias_data,
    params.activation, activation_state_data, scratchpad_data, output_data);
}

static inline void SetupScratchpadTensor(
  const luci_interpreter::DataType &input_data_type,
  const luci_interpreter::DataType &weight_feature_data_type,
  luci_interpreter::Tensor *scratchpad_1, luci_interpreter::Tensor *scratchpad_2,
  luci_interpreter::Tensor *scratchpad_3, luci_interpreter::Tensor *scratchpad_4,
  luci_interpreter::Tensor *scratchpad_5, luci_interpreter::Tensor *scratchpad_6,
  const luci_interpreter::Shape input_shape, const luci_interpreter::Shape weight_time_shape,
  const int32_t batch_size, const int32_t num_filters, const int32_t num_units)
{

  if (input_data_type == luci_interpreter::DataType::FLOAT32 &&
      (weight_feature_data_type == luci_interpreter::DataType::S8 ||
       weight_feature_data_type == luci_interpreter::DataType::U8))
  {
    (void)input_shape;
    (void)weight_time_shape;
    (void)scratchpad_3;
    (void)scratchpad_4;
    (void)scratchpad_5;
    (void)scratchpad_6;

    throw std::runtime_error("Hybrid type is not currently supported for mcu platform");
  }

  // Resize scratchpad_1 tensor
  scratchpad_1->resize({batch_size, num_filters});

  if (input_data_type == luci_interpreter::DataType::S8)
  {
    // Resize scratchpad_2 for full_integer op
    scratchpad_2->resize({batch_size, num_units});
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_SVDF_H
