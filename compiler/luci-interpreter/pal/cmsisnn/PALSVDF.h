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

#include <arm_nn_types.h>
#include <arm_nnfunctions.h>

namespace luci_interpreter_pal
{
static inline void
IntegerSVDF(const TfLiteSVDFParams &params, const tflite::RuntimeShape &input_shape,
            const int8_t *input_data, const tflite::RuntimeShape &weight_feature_shape,
            const int8_t *weight_feature_data, const tflite::RuntimeShape &weight_time_shape,
            const int16_t *weight_time_data, const tflite::RuntimeShape &bias_shape,
            const int32_t *bias_data, int16_t *activation_state_data,
            const tflite::RuntimeShape &output_shape, int8_t *output_data, int32_t *scratchpad_data,
            int32_t *output_temp_data, int32_t scale_1_a, int32_t scale_1_b, int32_t scale_2_a,
            int32_t scale_2_b, int32_t input_zp, int32_t output_zp)
{
  const int32_t rank = params.rank;
  const int32_t batch_size = input_shape.Dims(0);
  const int32_t num_filters = weight_feature_shape.Dims(0);
  const int32_t memory_size = weight_time_shape.Dims(1);

  cmsis_nn_dims input_dims;
  input_dims.n = input_shape.Dims(0);
  input_dims.h = input_shape.Dims(1);

  cmsis_nn_dims weights_feature_dims;
  weights_feature_dims.n = weight_feature_shape.Dims(0);
  weights_feature_dims.h = weight_feature_shape.Dims(1);

  cmsis_nn_dims weights_time_dims;
  weights_time_dims.n = weight_time_shape.Dims(0);
  weights_time_dims.h = weight_time_shape.Dims(1);

  cmsis_nn_dims bias_dims;
  bias_dims.n = bias_shape.Dims(0);

  cmsis_nn_dims state_dims;
  state_dims.n = batch_size;
  state_dims.h = memory_size * num_filters;

  cmsis_nn_dims output_dims;
  output_dims.n = output_shape.Dims(0);
  output_dims.h = output_shape.Dims(1);

  cmsis_nn_svdf_params svdf_params;
  svdf_params.rank = params.rank;
  svdf_params.input_offset = input_zp;
  svdf_params.output_offset = output_zp;

  svdf_params.input_activation.min = INT16_MIN;
  svdf_params.input_activation.max = INT16_MAX;

  svdf_params.output_activation.min = INT8_MIN;
  svdf_params.output_activation.max = INT8_MAX;

  cmsis_nn_per_tensor_quant_params in_quant_params;
  in_quant_params.multiplier = scale_1_a;
  in_quant_params.shift = scale_1_b;

  cmsis_nn_per_tensor_quant_params out_quant_params;
  out_quant_params.multiplier = scale_2_a;
  out_quant_params.shift = scale_2_b;

  cmsis_nn_context scratch_ctx;
  scratch_ctx.buf = scratchpad_data;

  cmsis_nn_context scratch_output_ctx;
  scratch_output_ctx.buf = output_temp_data;

  arm_svdf_s8(&scratch_ctx, &scratch_output_ctx, &svdf_params, &in_quant_params, &out_quant_params,
              &input_dims, input_data, &state_dims, activation_state_data, &weights_feature_dims,
              weight_feature_data, &weights_time_dims, weight_time_data, &bias_dims, bias_data,
              &output_dims, output_data);
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
  if (input_data_type == loco::DataType::FLOAT32 &&
      (weight_feature_data_type == loco::DataType::S8 ||
       weight_feature_data_type == loco::DataType::U8))
  {
    (void)input_shape;
    (void)weight_time_shape;
    (void)scratchpad_3;
    (void)scratchpad_4;
    (void)scratchpad_5;
    (void)scratchpad_6;

    throw std::runtime_error("Hybrid type is not supported for cmsisnn");
  }

  // Resize scratchpad_1 tensor
  scratchpad_1->resize({batch_size, num_filters});

  if (input_data_type == loco::DataType::S8)
  {
    // Resize scratchpad_2 for full_integer op
    scratchpad_2->resize({batch_size, num_units});
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_SVDF_H
