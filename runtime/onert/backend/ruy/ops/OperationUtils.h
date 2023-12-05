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
#include <ir/DataType.h>
#include <ir/Padding.h>
#include <util/CalculateActivationRange.h>

#include <ruy/Shape.h>
#include <ruy/Types.h>

#include <limits>

using OperandType = onert::ir::DataType;
using namespace onert::util;

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

  assert(tensor->layout() == ir::Layout::NHWC || tensor->layout() == ir::Layout::UNKNOWN);

  auto rank = shape.rank();
  nnfw::ruy::Shape ret(rank);
  auto data = ret.DimsData();
  for (int i = 0; i < rank; ++i)
  {
    data[i] = shape.dim(i);
  }
  return ret;
}

inline nnfw::ruy::FusedActivationFunctionType convertActivationType(const ir::Activation activation)
{
  switch (activation)
  {
    case ir::Activation::NONE:
      return nnfw::ruy::FusedActivationFunctionType::kNone;
    case ir::Activation::RELU:
      return nnfw::ruy::FusedActivationFunctionType::kRelu;
    case ir::Activation::RELU1:
      return nnfw::ruy::FusedActivationFunctionType::kRelu1;
    case ir::Activation::RELU6:
      return nnfw::ruy::FusedActivationFunctionType::kRelu6;
    case ir::Activation::TANH:
      return nnfw::ruy::FusedActivationFunctionType::kTanh;
    case ir::Activation::SIGMOID:
      return nnfw::ruy::FusedActivationFunctionType::kSigmoid;
    default:
      throw std::runtime_error{"RUY backend: Cannot convert activation type"};
  }
}

nnfw::ruy::PaddingType getPaddingType(ir::PaddingType ir_padding_type);

} // namespace ops
} // namespace ruy
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_RUY_OPS_OPERATION_UTILS_H__
