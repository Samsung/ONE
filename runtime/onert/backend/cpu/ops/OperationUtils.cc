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
namespace ops
{

uint32_t getNumberOfDimensions(const IPortableTensor *tensor)
{
  assert(tensor);
  return tensor->getShape().rank();
}

uint32_t getNumberOfElements(const IPortableTensor *tensor)
{
  assert(tensor);
  uint32_t count = 1;
  auto shape = tensor->getShape();
  for (int i = 0; i < shape.rank(); i++)
  {
    count *= shape.dim(i);
  }
  return count;
}

uint32_t getSizeOfDimension(const IPortableTensor *tensor, uint32_t dimensionIdx)
{
  assert(tensor);
  auto shape = tensor->getShape();
  if (dimensionIdx >= static_cast<uint32_t>(shape.rank()))
  {
    // TODO, log the error
    return 0;
  }
  return shape.dim(dimensionIdx);
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

void GetQuantizedConvolutionMultipliersAndShifts(
  float input_scale, float output_scale, const float *filter_scales, size_t filter_scales_size,
  int num_channels, std::vector<int32_t> &per_channel_output_multiplier,
  std::vector<int> &per_channel_output_shift)
{
  // Originates from tflite's PopulateConvolutionQuantizationParams()
  per_channel_output_multiplier.resize(num_channels);
  per_channel_output_shift.resize(num_channels);

  const bool is_per_channel = filter_scales_size > 1;
  auto per_channel_multiplier = per_channel_output_multiplier.data();
  auto per_channel_shift = per_channel_output_shift.data();
  for (int i = 0; i < num_channels; ++i)
  {
    // If per-tensor quantization parameter is specified, broadcast it along the
    // quantization dimension (channels_out).
    const float scale = is_per_channel ? filter_scales[i] : filter_scales[0];
    const double filter_scale = static_cast<double>(scale);
    const double effective_output_scale =
      static_cast<double>(input_scale) * filter_scale / static_cast<double>(output_scale);
    int32_t significand;
    int channel_shift;
    QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
    per_channel_multiplier[i] = significand;
    per_channel_shift[i] = channel_shift;
  }
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

void CalculateActivationRangeQuantized(ir::Activation activation, const IPortableTensor *output,
                                       int32_t *act_min, int32_t *act_max)
{
  int32_t qmin = 0;
  int32_t qmax = 0;

  switch (output->data_type())
  {
    case OperandType::QUANT_UINT8_ASYMM:
      qmin = std::numeric_limits<uint8_t>::min();
      qmax = std::numeric_limits<uint8_t>::max();
      break;
    case OperandType::QUANT_INT8_ASYMM:
    case OperandType::QUANT_INT8_SYMM:
      qmin = std::numeric_limits<int8_t>::min();
      qmax = std::numeric_limits<int8_t>::max();
      break;
    default:
      throw std::runtime_error("CalculateActivationRangeQuantized: Not supported operand type.");
  }

  const auto scale = output->data_scale();
  const auto zero_point = output->data_zero_point();
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
    throw std::runtime_error{"Unsupported fused activation function."};
  }
}

bool HaveSameShapes(const IPortableTensor *input1, const IPortableTensor *input2)
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

  auto shape1 = input1->getShape();
  auto shape2 = input2->getShape();
  for (uint32_t i = 0; i < getNumberOfDimensions(input1); i++)
    if (shape1.dim(i) != shape2.dim(i))
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

uint32_t sizeOfData(OperandType type, const std::vector<int32_t> &dimensions)
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
    case OperandType::QUANT_UINT8_ASYMM:
    case OperandType::QUANT_INT8_SYMM:
      size = 1;
      break;
    case OperandType::INT64:
      size = 8;
      break;
    default:
      throw std::runtime_error("Not supported operand type.");
      break;
  }

  for (auto &&d : dimensions)
  {
    assert(d >= 0);
    size *= static_cast<uint32_t>(d);
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

std::vector<int32_t> getReducerAxes(const IPortableTensor *axes)
{
  std::vector<int32_t> ret;

  auto axes_vals = (axes->getShape().rank() == 0) ? 1 : axes->getShape().dim(0);
  assert(axes->layout() == ir::Layout::NHWC || axes->layout() == ir::Layout::UNKNOWN);
  assert(static_cast<size_t>(axes_vals) == axes->getShape().num_elements());
  switch (axes->data_type())
  {
    case ir::DataType::INT32:
    {
      for (int i = 0; i < axes_vals; ++i)
        ret.emplace_back(*(getBuffer<int32_t>(axes) + i));
      break;
    }
    case ir::DataType::INT64:
    {
      for (int i = 0; i < axes_vals; ++i)
        ret.emplace_back(*(getBuffer<int64_t>(axes) + i));
      break;
    }
    default:
      throw std::runtime_error("getReducerAxes: Not supported data type");
      break;
  }
  return ret;
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
