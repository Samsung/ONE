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

#ifndef __ONERT_BACKEND_XNNPACK_OPS_OPERATION_UTILS_H__
#define __ONERT_BACKEND_XNNPACK_OPS_OPERATION_UTILS_H__

// duplicated from cpu/ops/OperationUtils.h
#include <ir/InternalType.h>
#include <ir/Padding.h>
#include <ir/DataType.h>

namespace onert
{
namespace backend
{
namespace xnnpack
{
namespace ops
{

using OperandType = ir::DataType;

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
    throw std::runtime_error{"Unsupported fused activation function"};
  }
}

} // namespace ops
} // namespace xnnpack
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_XNNPACK_OPS_OPERATION_UTILS_H__
