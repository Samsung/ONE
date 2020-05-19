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

#include "kernels/Utils.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

void calculateActivationRange(Activation activation, float *activation_min, float *activation_max)
{
  switch (activation)
  {
    case Activation::NONE:
      *activation_min = std::numeric_limits<float>::lowest();
      *activation_max = std::numeric_limits<float>::max();
      break;
    case Activation::RELU:
      *activation_min = 0;
      *activation_max = std::numeric_limits<float>::max();
      break;
    case Activation::RELU_N1_TO_1:
      *activation_min = -1;
      *activation_max = 1;
      break;
    case Activation::RELU6:
      *activation_min = 0;
      *activation_max = 6;
      break;
    default:
      throw std::runtime_error("Unsupported activation.");
  }
}

static void calculateActivationRangeQuantizedImpl(Activation activation, int32_t qmin, int32_t qmax,
                                                  const Tensor *output, int32_t *activation_min,
                                                  int32_t *activation_max)
{
  const float scale = output->scale();
  const int32_t zero_point = output->zero_point();

  auto quantize = [scale, zero_point](float x) {
    return zero_point + static_cast<int32_t>(std::round(x / scale));
  };

  switch (activation)
  {
    case Activation::NONE:
      *activation_min = qmin;
      *activation_max = qmax;
      break;
    case Activation::RELU:
      *activation_min = std::max(qmin, quantize(0.0f));
      *activation_max = qmax;
      break;
    case Activation::RELU_N1_TO_1:
      *activation_min = std::max(qmin, quantize(-1.0f));
      *activation_max = std::min(qmax, quantize(1.0f));
      break;
    case Activation::RELU6:
      *activation_min = std::max(qmin, quantize(0.0f));
      *activation_max = std::min(qmax, quantize(6.0f));
      break;
    default:
      throw std::runtime_error("Unsupported activation.");
  }
}

void calculateActivationRangeQuantized(Activation activation, const Tensor *output,
                                       int32_t *activation_min, int32_t *activation_max)
{
  int32_t qmin{};
  int32_t qmax{};
  switch (output->element_type())
  {
    case DataType::U8:
      qmin = std::numeric_limits<uint8_t>::min();
      qmax = std::numeric_limits<uint8_t>::max();
      break;
    case DataType::S8:
      qmin = std::numeric_limits<int8_t>::min();
      qmax = std::numeric_limits<int8_t>::max();
      break;
    case DataType::S16:
      qmin = std::numeric_limits<int16_t>::min();
      qmax = std::numeric_limits<int16_t>::max();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }

  calculateActivationRangeQuantizedImpl(activation, qmin, qmax, output, activation_min,
                                        activation_max);
}

} // namespace kernels
} // namespace luci_interpreter
