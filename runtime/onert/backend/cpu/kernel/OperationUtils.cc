/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
namespace cpu
{
namespace kernel
{

uint32_t getNumberOfDimensions(const ITensor *tensor)
{
  assert(tensor);
  return tensor->num_dimensions();
}

uint32_t getNumberOfElements(const ITensor *tensor)
{
  assert(tensor);
  uint32_t count = 1;
  for (size_t i = 0; i < tensor->num_dimensions(); i++)
  {
    count *= tensor->dimension(i);
  }
  return count;
}

uint32_t getSizeOfDimension(const ITensor *tensor, uint32_t dimensionIdx)
{
  assert(tensor);
  if (dimensionIdx >= tensor->num_dimensions())
  {
    // TODO, log the error
    return 0;
  }
  return tensor->dimension(dimensionIdx);
}

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

void GetQuantizedConvolutionMultiplier(const ITensor *input, const ITensor *filter,
                                       const ITensor *bias, const ITensor *output,
                                       double *multiplier)
{
  const double input_product_scale = input->data_scale() * filter->data_scale();
  const double bias_scale = bias->data_scale();
  const double output_scale = output->data_scale();
  // The following conditions must be guaranteed by the training pipeline.
  UNUSED_RELEASE(bias_scale);
  assert(std::abs(input_product_scale - bias_scale) <=
         1e-6 * std::min(input_product_scale, bias_scale));
  assert(input_product_scale >= 0);
  assert(input_product_scale < output_scale);
  *multiplier = input_product_scale / output_scale;
}

void QuantizeMultiplierGreaterThanOne(double double_multiplier, int32_t *quantized_multiplier,
                                      int *left_shift)
{
  assert(double_multiplier > 1.);
  const double q = std::frexp(double_multiplier, left_shift);
  int64_t q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
  assert(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31))
  {
    q_fixed /= 2;
    ++*left_shift;
  }
  assert(*left_shift >= 0);
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void CalculateActivationRangeFloat(ir::Activation activation, float *activation_min,
                                   float *activation_max)
{
  if (activation == ir::Activation::RELU)
  {
    *activation_min = 0.f;
    *activation_max = std::numeric_limits<float>::max();
  }
  else if (activation == ir::Activation::RELU6)
  {
    *activation_min = 0.f;
    *activation_max = 6.f;
  }
  else if (activation == ir::Activation::RELU1)
  {
    *activation_min = -1.f;
    *activation_max = 1.f;
  }
  else if (activation == ir::Activation::SIGMOID)
  {
    *activation_min = 0.f;
    *activation_max = 1.f;
  }
  else if (activation == ir::Activation::NONE)
  {
    *activation_min = std::numeric_limits<float>::lowest();
    *activation_max = std::numeric_limits<float>::max();
  }
  else
  {
    std::cout << "Unsupported fused activation function." << std::endl;
  }
}

void CalculateActivationRangeUint8(ir::Activation activation, const ITensor *output,
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

bool HaveSameShapes(const ITensor *input1, const ITensor *input2)
{
  if (input1 == input2)
    return true;
  if (input2 == NULL || input2 == NULL)
    return false;

  if (input1 == NULL)
  {
    return (getNumberOfDimensions(input2) == 0);
  }

  if (getNumberOfDimensions(input1) != getNumberOfDimensions(input2))
    return false;

  for (uint32_t i = 0; i < getNumberOfDimensions(input1); i++)
    if (input1->dimension(i) != input2->dimension(i))
      return false;

  return true;
}

int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift)
{
  const double max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
                                    (1ll << (31 - input_integer_bits)) / (1ll << input_left_shift);
  // Tighten bound using floor.  Suppose that we could use the exact value.
  // After scaling the difference, the result would be at the maximum.  Thus we
  // must ensure that our value has lower magnitude.
  return static_cast<int32_t>(std::floor(max_input_rescaled));
}

uint32_t sizeOfData(OperandType type, const std::vector<uint32_t> &dimensions)
{
  uint32_t size = 4;

  switch (type)
  {
    case OperandType::FLOAT32:
    case OperandType::INT32:
    case OperandType::UINT32:
      size = 4;
      break;
    case OperandType::BOOL8:
    case OperandType::QUANT8_ASYMM:
    case OperandType::QUANT8_SYMM:
      size = 1;
      break;
    default:
      throw std::runtime_error("Not supported operand type.");
      break;
  }

  for (auto d : dimensions)
  {
    size *= d;
  }

  return size;
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

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert
