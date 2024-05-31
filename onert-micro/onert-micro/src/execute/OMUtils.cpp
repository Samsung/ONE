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

#include "execute/OMUtils.h"
#include <cmath>

using namespace onert_micro::execute;
using namespace onert_micro;

void onert_micro::execute::quantizeMultiplier(double double_multiplier,
                                              int32_t *quantized_multiplier, int *shift)
{
  if (double_multiplier == 0.0)
  {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }

  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(std::round(q * (int64_t(1) << 31)));

  if (q_fixed == (int64_t(1) << 31))
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

void onert_micro::execute::quantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                                               int32_t *quantized_multiplier,
                                                               int *left_shift)
{
  assert(double_multiplier < 1.0);
  assert(double_multiplier > 0.0);
  int shift;
  onert_micro::execute::quantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
  assert(shift <= 0);
  *left_shift = shift;
}

namespace
{
static void calculateActivationRangeQuantizedImpl(circle::ActivationFunctionType activation,
                                                  int32_t qmin, int32_t qmax, int32_t zero_point,
                                                  float scale, int32_t *activation_min,
                                                  int32_t *activation_max)
{
  assert(scale != 0.f);

  auto quantize = [scale, zero_point](float x) {
    return zero_point + static_cast<int32_t>(std::round(x / scale));
  };

  switch (activation)
  {
    case circle::ActivationFunctionType::ActivationFunctionType_NONE:
    case circle::ActivationFunctionType::ActivationFunctionType_TANH:
      *activation_min = qmin;
      *activation_max = qmax;
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU:
      *activation_min = std::max(qmin, quantize(0.0f));
      *activation_max = qmax;
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU_N1_TO_1:
      *activation_min = std::max(qmin, quantize(-1.0f));
      *activation_max = std::min(qmax, quantize(1.0f));
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU6:
      *activation_min = std::max(qmin, quantize(0.0f));
      *activation_max = std::min(qmax, quantize(6.0f));
      break;
    default:
      assert(false && "Unsupported activation.");
  }
}
} // namespace

void onert_micro::execute::calculateActivationRangeQuantized(
  circle::ActivationFunctionType activation, int32_t output_zero_point, float output_scale,
  circle::TensorType data_type, int32_t *activation_min, int32_t *activation_max)
{
  int32_t qmin;
  int32_t qmax;
  switch (data_type)
  {
    case circle::TensorType_UINT8:
      qmin = 0;
      qmax = std::numeric_limits<uint8_t>::max();
      break;
    case circle::TensorType_INT8:
      qmin = std::numeric_limits<int8_t>::min();
      qmax = std::numeric_limits<int8_t>::max();
      break;
    case circle::TensorType_INT16:
      // For now, assume that signed int16 type implies signed symmetric quantization.
      assert(output_zero_point == 0);
      qmin = std::numeric_limits<int16_t>::min();
      qmax = std::numeric_limits<int16_t>::max();
      break;
    default:
      assert(false && "Unsupported type.");
  }

  calculateActivationRangeQuantizedImpl(activation, qmin, qmax, output_zero_point, output_scale,
                                        activation_min, activation_max);
}
