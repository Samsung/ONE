/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OperationUtils.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace onert
{
namespace backend
{
namespace ruy
{
namespace ops
{

void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift)
{
  if (double_multiplier == 0.)
  {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));

  assert(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31))
  {
    q_fixed /= 2;
    ++*shift;
  }
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void GetQuantizedConvolutionMultiplier(const IPortableTensor *input, const IPortableTensor *filter,
                                       const IPortableTensor *bias, const IPortableTensor *output,
                                       double *multiplier)
{
  const double input_product_scale = input->data_scale() * filter->data_scale();
  const double bias_scale = (bias != nullptr) ? bias->data_scale() : input_product_scale;
  const double output_scale = output->data_scale();
  // The following conditions must be guaranteed by the training pipeline.
  UNUSED_RELEASE(bias_scale);
  assert(std::abs(input_product_scale - bias_scale) <=
         1e-6 * std::min(input_product_scale, bias_scale));
  assert(input_product_scale >= 0);
  assert(input_product_scale < output_scale);
  *multiplier = input_product_scale / output_scale;
}

void CalculateActivationRangeUint8(ir::Activation activation, const IPortableTensor *output,
                                   int32_t *act_min, int32_t *act_max)
{
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();
  const auto scale = output->data_scale();
  const auto zero_point = output->data_offset();
  auto quantize = [scale, zero_point](float f) {
    return zero_point + static_cast<int32_t>(std::round(f / scale));
  };
  if (activation == ir::Activation::RELU)
  {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = qmax;
  }
  else if (activation == ir::Activation::RELU6)
  {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = std::min(qmax, quantize(6.0));
  }
  else if (activation == ir::Activation::RELU1)
  {
    *act_min = std::max(qmin, quantize(-1.0));
    *act_max = std::min(qmax, quantize(1.0));
  }
  else if (activation == ir::Activation::SIGMOID)
  {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = std::min(qmax, quantize(1.0));
  }
  else if (activation == ir::Activation::NONE)
  {
    *act_min = qmin;
    *act_max = qmax;
  }
  else
  {
    std::cout << "Unsupported fused activation function." << std::endl;
  }
}

nnfw::cker::PaddingType getPaddingType(ir::PaddingType ir_padding_type)
{
  switch (ir_padding_type)
  {
    case ir::PaddingType::EXPLICIT:
      return nnfw::cker::PaddingType::kNone;
    case ir::PaddingType::SAME:
      return nnfw::cker::PaddingType::kSame;
    case ir::PaddingType::VALID:
      return nnfw::cker::PaddingType::kValid;
    default:
      throw std::runtime_error("Wrong padding type.");
      break;
  }
}

} // namespace ops
} // namespace ruy
} // namespace backend
} // namespace onert
