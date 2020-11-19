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

#ifndef __ONERT_BACKEND_RUY_OPS_OPERATION_UTILS_H__
#define __ONERT_BACKEND_RUY_OPS_OPERATION_UTILS_H__

#include <backend/IPortableTensor.h>

#include <ruy/Shape.h>
#include <ruy/Types.h>
#include <iostream>
#include <ir/DataType.h>
#include <ir/InternalType.h>
#include <ir/Padding.h>

#include <limits>

using OperandType = onert::ir::DataType;

namespace onert
{
namespace backend
{
namespace ruy
{
namespace ops
{

inline nnfw::ruy::Shape getTensorShape(const IPortableTensor *tensor)
{
  if (tensor == nullptr)
    return nnfw::ruy::Shape();

  const ir::Shape &shape = tensor->get_info().shape();

  assert(tensor->layout() == ir::Layout::NHWC);

  auto rank = shape.rank();
  nnfw::ruy::Shape ret(rank);
  auto data = ret.DimsData();
  for (int i = 0; i < rank; ++i)
  {
    data[i] = shape.dim(i);
  }
  return ret;
}

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

nnfw::ruy::PaddingType getPaddingType(ir::PaddingType ir_padding_type);

} // namespace ops
} // namespace ruy
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_RUY_OPS_OPERATION_UTILS_H__
