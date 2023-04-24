/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_PAL_SOFTMAX_H
#define LUCI_INTERPRETER_PAL_SOFTMAX_H

#include <tensorflow/lite/kernels/internal/reference/softmax.h>
#include <arm_nnfunctions.h>

namespace luci_interpreter_pal
{
static inline void PopulateSoftmaxLookupTable(tflite::SoftmaxParams *data, float input_scale,
                                              float beta)
{
  // Do nothing for mcu
  (void)data;
  (void)input_scale;
  (void)beta;
}

static inline void InitializeParams(tflite::SoftmaxParams *params, float input_scale, float beta)
{
  int32 input_beta_multiplier;
  int input_beta_left_shift;
  static const int kScaledDiffIntegerBits = 5;
  tflite::PreprocessSoftmaxScaling(beta, input_scale, kScaledDiffIntegerBits,
                                   &input_beta_multiplier, &input_beta_left_shift);

  params->input_multiplier = input_beta_multiplier;
  params->input_left_shift = input_beta_left_shift;
  params->diff_min =
    -tflite::CalculateInputRadius(kScaledDiffIntegerBits, params->input_left_shift);
}

template <typename T>
static inline void Softmax(const tflite::SoftmaxParams &params,
                           const tflite::RuntimeShape &input_shape, const T *input_data,
                           const tflite::RuntimeShape &output_shape, T *output_data)
{
  // MARK: At this moment this operation doesn't support on mcu
  assert(false && "Softmax NYI");
  (void)params;
  (void)input_shape;
  (void)input_data;
  (void)output_shape;
  (void)output_data;
}

template <>
inline void Softmax<int8_t>(const tflite::SoftmaxParams &params,
                            const tflite::RuntimeShape &input_shape, const int8_t *input_data,
                            const tflite::RuntimeShape &output_shape, int8_t *output_data)
{
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size = tflite::MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth = tflite::MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  const int32_t mult = params.input_multiplier;
  const int32_t shift = params.input_left_shift;
  const int32_t diff_min = params.diff_min;

  arm_softmax_s8(input_data, outer_size, depth, mult, shift, diff_min, output_data);
}
} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_SOFTMAX_H
