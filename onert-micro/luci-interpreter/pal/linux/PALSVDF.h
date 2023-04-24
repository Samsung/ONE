/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
  tflite::reference_ops::EvalIntegerSVDF(&params, input_shape, input_data, weight_feature_shape,
                                         weight_feature_data, weight_time_shape, weight_time_data,
                                         bias_shape, bias_data, activation_state_data, output_shape,
                                         output_data, scratchpad_data, output_temp_data, scale_1_a,
                                         scale_1_b, scale_2_a, scale_2_b, input_zp, output_zp);
}
static inline void
FloatSVDF(const TfLiteSVDFParams &params, const tflite::RuntimeShape &input_shape,
          const float *input_data, const tflite::RuntimeShape &weight_feature_shape,
          const float *weight_feature_data, const tflite::RuntimeShape &weight_time_shape,
          const float *weight_time_data, const tflite::RuntimeShape &bias_shape,
          const float *bias_data, float *scratchpad_data, float *activation_state_data,
          const tflite::RuntimeShape &output_shape, float *output_data)
{
  tflite::reference_ops::EvalFloatSVDF(&params, input_shape, input_data, weight_feature_shape,
                                       weight_feature_data, weight_time_shape, weight_time_data,
                                       bias_shape, bias_data, scratchpad_data,
                                       activation_state_data, output_shape, output_data);
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

    assert(false && "Hybrid type is not currently supported for linux platform");
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
