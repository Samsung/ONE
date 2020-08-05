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

#ifndef __NNFW_SUPPORT_NNAPI_OPERATION_UTILS_H__
#define __NNFW_SUPPORT_NNAPI_OPERATION_UTILS_H__

#include <backend/IPortableTensor.h>

#include <cker/Shape.h>
#include <cker/Types.h>
#include <iostream>
#include <ir/DataType.h>
#include <ir/InternalType.h>
#include <ir/Operand.h>
#include <ir/Padding.h>

#include <limits>
#include <vector>

using OperandType = onert::ir::DataType;

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

union DataPtr {
  uint8_t *u8;
  int8_t *i8;
  uint32_t *u32;
  int32_t *i32;
  bool *b;
  float *f;
  int64_t *i64;
  void *v;
};

union ConstDataPtr {
  const uint8_t *u8;
  const int8_t *i8;
  const uint32_t *u32;
  const int32_t *i32;
  const bool *b;
  const float *f;
  const int64_t *i64;
  const void *v;
};

uint32_t getNumberOfDimensions(const IPortableTensor *tensor);

uint32_t getNumberOfElements(const IPortableTensor *tensor);

uint32_t getSizeOfDimension(const IPortableTensor *tensor, uint32_t dimensionIdx);

inline nnfw::cker::Shape getExtendedTensorShape(const IPortableTensor *tensor)
{
  assert(tensor);
  const int32_t extended_rank = 4;
  int32_t raw_shape[extended_rank];
  uint32_t src = extended_rank - tensor->num_dimensions();
  for (uint32_t i = 0; i < extended_rank; ++i)
  {
    if (i < src)
    {
      raw_shape[i] = 1;
    }
    else
    {
      raw_shape[i] = tensor->dimension(i - src);
    }
  }

  return nnfw::cker::Shape(extended_rank, raw_shape);
}

inline nnfw::cker::Shape getTensorShape(const IPortableTensor *tensor)
{
  if (tensor == nullptr)
    return nnfw::cker::Shape();

  assert(tensor->layout() == ir::Layout::NHWC);
  constexpr int kMaxSmallSize = 8;
  int32_t raw_shape_small[kMaxSmallSize];
  std::vector<int32_t> raw_shape_vec;
  auto rank = tensor->num_dimensions();
  int32_t *data = nullptr;
  if (rank > kMaxSmallSize)
  {
    raw_shape_vec.resize(rank);
    data = raw_shape_vec.data();
  }
  else
  {
    data = raw_shape_small;
  }

  for (uint32_t i = 0; i < rank; ++i)
  {
    data[i] = tensor->dimension(i);
  }
  return nnfw::cker::Shape(rank, data);
}

inline nnfw::cker::FusedActivationFunctionType
convertActivationType(const ir::Activation activation)
{
  switch (activation)
  {
    case ir::Activation::NONE:
      return nnfw::cker::FusedActivationFunctionType::kNone;
    case ir::Activation::RELU:
      return nnfw::cker::FusedActivationFunctionType::kRelu;
    case ir::Activation::RELU1:
      return nnfw::cker::FusedActivationFunctionType::kRelu1;
    case ir::Activation::RELU6:
      return nnfw::cker::FusedActivationFunctionType::kRelu6;
    default:
      throw std::runtime_error{"CPU backend: Cannot convert activation type"};
  }
}

inline int32_t getAxis(uint32_t rank, int32_t axis, ir::Layout frontend_layout)
{
  auto ret = axis;

  if (axis < 0)
  {
    ret += rank;
  }

  // NCHW -> NHWC
  if (frontend_layout == ir::Layout::NCHW)
  {
    int32_t permutation[4] = {0, 3, 1, 2};
    ret = permutation[ret];
  }

  return ret;
}

void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift);

void GetQuantizedConvolutionMultiplier(const IPortableTensor *inputDescr,
                                       const IPortableTensor *filterDescr,
                                       const IPortableTensor *biasDescr,
                                       const IPortableTensor *outputDescr, double *multiplier);

void QuantizeMultiplierGreaterThanOne(double double_multiplier, int32_t *quantized_multiplier,
                                      int *left_shift);

template <typename T>
void CalculateActivationRange(ir::Activation activation, T *activation_min, T *activation_max)
{
  if (activation == ir::Activation::RELU)
  {
    *activation_min = 0;
    *activation_max = std::numeric_limits<T>::max();
  }
  else if (activation == ir::Activation::RELU6)
  {
    *activation_min = 0;
    *activation_max = 6;
  }
  else if (activation == ir::Activation::RELU1)
  {
    *activation_min = -1;
    *activation_max = 1;
  }
  else if (activation == ir::Activation::SIGMOID)
  {
    *activation_min = 0;
    *activation_max = 1;
  }
  else if (activation == ir::Activation::NONE)
  {
    *activation_min = std::numeric_limits<T>::lowest();
    *activation_max = std::numeric_limits<T>::max();
  }
  else
  {
    std::cout << "Unsupported fused activation function." << std::endl;
  }
}

void CalculateActivationRangeUint8(ir::Activation activation, const IPortableTensor *output,
                                   int32_t *act_min, int32_t *act_max);

bool HaveSameShapes(const IPortableTensor *input1, const IPortableTensor *input2);

int32_t CalculateInputRadius(int input_integer_bits, int input_left_shift);

uint32_t sizeOfData(OperandType type, const std::vector<int32_t> &dimensions);

nnfw::cker::PaddingType getPaddingType(ir::PaddingType ir_padding_type);

std::vector<int32_t> getReducerAxes(const IPortableTensor *axes);

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __NNFW_SUPPORT_NNAPI_OPERATION_UTILS_H__
