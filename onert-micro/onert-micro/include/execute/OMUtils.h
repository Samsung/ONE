/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_EXECUTE_UTILS_H
#define ONERT_MICRO_EXECUTE_UTILS_H

#include "OMStatus.h"
#include "core/reader/OMCircleReader.h"
#include "core/OMRuntimeShape.h"

namespace onert_micro
{
namespace execute
{

template <typename T>
OMStatus calculateActivationRange(circle::ActivationFunctionType activation, T *activation_min,
                                  T *activation_max)
{
  switch (activation)
  {
    case circle::ActivationFunctionType::ActivationFunctionType_NONE:
      *activation_min = std::numeric_limits<T>::lowest();
      *activation_max = std::numeric_limits<T>::max();
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU:
      *activation_min = 0;
      *activation_max = std::numeric_limits<T>::max();
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU_N1_TO_1:
      *activation_min = -1;
      *activation_max = 1;
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU6:
      *activation_min = 0;
      *activation_max = 6;
      break;
    default:
      assert(false && "Unsupported activation.");
      return UnsupportedActivation;
  }

  return Ok;
}

inline double getQuantizedConvolutionMultipler(float input_scale, float filter_scale,
                                               float output_scale)
{
  const double input_product_scale = static_cast<double>(input_scale * filter_scale);

  assert(input_product_scale >= 0);

  assert(output_scale != 0.f);

  return input_product_scale / static_cast<double>(output_scale);
}

void quantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift);

void calculateActivationRangeQuantized(circle::ActivationFunctionType activation,
                                       int32_t output_zero_point, float output_scale,
                                       circle::TensorType data_type, int32_t *activation_min,
                                       int32_t *activation_max);

inline int computeOutSize(circle::Padding padding, int image_size, int filter_size, int stride,
                          int dilation_rate = 1)
{
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;

  if (stride == 0)
    return 0;

  switch (padding)
  {
    case circle::Padding_SAME:
      return (image_size + stride - 1) / stride;
    case circle::Padding_VALID:
      return (image_size + stride - effective_filter_size) / stride;
    default:
      return 0;
  }
}

inline int computePadding(int32_t stride, int32_t dilation_rate, int32_t in_size,
                          int32_t filter_size, int32_t out_size)
{
  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  int32_t padding = ((out_size - 1) * stride + effective_filter_size - in_size) / 2;
  return padding > 0 ? padding : 0;
}

inline void computePaddingHeightWidth(int32_t stride_height, int32_t stride_width,
                                      int32_t dilation_rate_height, int32_t dilation_rate_width,
                                      int32_t in_height, int32_t in_width, int32_t filter_height,
                                      int32_t filter_width, circle::Padding padding,
                                      int32_t *padding_h, int32_t *padding_w)
{

  int out_width =
    computeOutSize(padding, in_width, filter_width, stride_width, dilation_rate_width);
  int out_height =
    computeOutSize(padding, in_height, filter_height, stride_height, dilation_rate_height);

  *padding_h =
    computePadding(stride_height, dilation_rate_height, in_height, filter_height, out_height);

  *padding_w = computePadding(stride_width, dilation_rate_width, in_width, filter_width, out_width);
}

} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_UTILS_H
