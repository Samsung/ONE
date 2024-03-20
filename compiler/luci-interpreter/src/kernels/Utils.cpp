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

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

TfLiteFusedActivation getTfLiteActivation(Activation activation)
{
  switch (activation)
  {
    case luci::FusedActFunc::RELU:
      return kTfLiteActRelu;
    case luci::FusedActFunc::RELU6:
      return kTfLiteActRelu6;
    case luci::FusedActFunc::RELU_N1_TO_1:
      return kTfLiteActReluN1To1;
    case luci::FusedActFunc::TANH:
      return kTfLiteActTanh;
    case luci::FusedActFunc::SIGN_BIT:
      return kTfLiteActSignBit;
    case luci::FusedActFunc::NONE:
      return kTfLiteActNone;
    default:
      throw std::runtime_error("Unsupported activation type");
  }
}

template <typename T>
void calculateActivationRange(Activation activation, T *activation_min, T *activation_max)
{
  switch (activation)
  {
    case Activation::NONE:
      *activation_min = std::numeric_limits<T>::lowest();
      *activation_max = std::numeric_limits<T>::max();
      break;
    case Activation::RELU:
      *activation_min = 0;
      *activation_max = std::numeric_limits<T>::max();
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

template void calculateActivationRange(Activation activation, float *activation_min,
                                       float *activation_max);
template void calculateActivationRange(Activation activation, int32_t *activation_min,
                                       int32_t *activation_max);
template void calculateActivationRange(Activation activation, int64_t *activation_min,
                                       int64_t *activation_max);

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
    case Activation::TANH:
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
  assert(output->zero_points().size() == 1);
  int32_t qmin{};
  int32_t qmax{};
  switch (output->element_type())
  {
    case DataType::U4:
      qmin = 0;
      qmax = 15;
      break;
    case DataType::U8:
      qmin = 0;
      qmax = std::numeric_limits<uint8_t>::max();
      break;
    case DataType::S4:
      qmin = -8;
      qmax = 7;
      break;
    case DataType::S8:
      qmin = -std::numeric_limits<int8_t>::max();
      qmax = std::numeric_limits<int8_t>::max();
      break;
    case DataType::S16:
      // For now, assume that signed int16 type implies signed symmetric quantization.
      assert(output->zero_point() == 0);
      qmin = -std::numeric_limits<int16_t>::max();
      qmax = std::numeric_limits<int16_t>::max();
      break;
    default:
      throw std::runtime_error("luci-intp (calculateActivationRangeQuantized) Unsupported type.");
  }

  calculateActivationRangeQuantizedImpl(activation, qmin, qmax, output, activation_min,
                                        activation_max);
}

void quantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift)
{
  if (double_multiplier == 0.0)
  {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }

  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(std::round(q * (INT64_C(1) << 31)));

  if (q_fixed == (INT64_C(1) << 31))
  {
    q_fixed /= 2;
    ++*shift;
  }
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (*shift < -31)
  {
    *shift = 0;
    q_fixed = 0;
  }
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void quantizeMultiplierSmallerThanOneExp(double double_multiplier, int32_t *quantized_multiplier,
                                         int *left_shift)
{
  assert(double_multiplier < 1.0);
  assert(double_multiplier > 0.0);
  int shift;
  quantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
  assert(shift <= 0);
  *left_shift = shift;
}

Shape calculateShapeForBroadcast(const Shape &input1_shape, const Shape &input2_shape)
{
  const int num_input1_dims = input1_shape.num_dims();
  const int num_input2_dims = input2_shape.num_dims();
  const int num_out_dims = std::max(num_input1_dims, num_input2_dims);
  Shape output_shape(num_out_dims);

  for (int i = 0; i < num_out_dims; ++i)
  {
    const int32_t input1_dim = i < num_input1_dims ? input1_shape.dim(num_input1_dims - i - 1) : 1;
    const int32_t input2_dim = i < num_input2_dims ? input2_shape.dim(num_input2_dims - i - 1) : 1;

    bool need_broadcast = input1_dim != input2_dim;
    bool can_broadcast = input1_dim == 1 || input2_dim == 1;
    LUCI_INTERPRETER_CHECK(!need_broadcast || can_broadcast);

    output_shape.dim(num_out_dims - i - 1) = std::max(input1_dim, input2_dim);
  }

  return output_shape;
}

} // namespace kernels
} // namespace luci_interpreter
