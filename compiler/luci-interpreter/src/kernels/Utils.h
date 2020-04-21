/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_KERNELS_UTILS_H
#define LUCI_INTERPRETER_KERNELS_UTILS_H

#include "core/KernelParams.h"
#include "core/Tensor.h"

#include <public/gemmlowp.h>

#include <cassert>
#include <cstdint>

namespace luci_interpreter
{
namespace kernels
{

inline int32_t computePadding(int32_t stride, int32_t dilation_rate, int32_t in_size,
                              int32_t filter_size, int32_t out_size)
{
  const int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  const int32_t padding = ((out_size - 1) * stride + effective_filter_size - in_size) / 2;
  return padding > 0 ? padding : 0;
}

inline int32_t computeOutputSize(Padding padding, int32_t image_size, int32_t filter_size,
                                 int32_t stride, int32_t dilation_rate = 1)
{
  const int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding)
  {
    case Padding::SAME:
      return (image_size + stride - 1) / stride;
    case Padding::VALID:
      return (image_size + stride - effective_filter_size) / stride;
    default:
      assert(false);
      return 0;
  }
}

void calculateActivationRange(Activation activation, float *activation_min, float *activation_max);

void calculateActivationRangeQuantized(Activation activation, const Tensor *output,
                                       int32_t *activation_min, int32_t *activation_max);

template <typename T> inline T activationFunctionWithMinMax(T x, T activation_min, T activation_max)
{
  return (x < activation_min) ? activation_min : (x > activation_max) ? activation_max : x;
}

inline int32_t offset(const Shape &shape, int32_t i0, int32_t i1, int32_t i2, int32_t i3)
{
  assert(shape.num_dims() == 4);
  assert(i0 >= 0 && i0 < shape.dim(0));
  return ((i0 * shape.dim(1) + i1) * shape.dim(2) + i2) * shape.dim(3) + i3;
}

void quantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift);

void quantizeMultiplierSmallerThanOneExp(double double_multiplier, int32_t *quantized_multiplier,
                                         int *left_shift);

inline int32_t multiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift)
{
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier), right_shift);
}

inline int32_t multiplyByQuantizedMultiplierSmallerThanOneExp(int32_t x,
                                                              int32_t quantized_multiplier,
                                                              int left_shift)
{
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, quantized_multiplier),
                             -left_shift);
}

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_UTILS_H
