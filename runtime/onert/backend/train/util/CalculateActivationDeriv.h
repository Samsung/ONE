/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_UTIL_CALCULATE_ACTIVATION_RANGE_H__
#define __ONERT_BACKEND_TRAIN_UTIL_CALCULATE_ACTIVATION_RANGE_H__

#include <limits>

#include <ir/InternalType.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace util
{

template <typename T>
void multiply_strided()
template <typename T>
void CalculateActivationDeriv(ir::Activation activation, const IPortableTensor *input, const IPortableTensor *output, const IPortableTensor *grad_input, IPortableTensor *grad_output)
{
  if (activation == ir::Activation::RELU)
  {
    grad_output = grad_input;
  }
  // else if (activation == ir::Activation::RELU6)
  // {
  //   *activation_min = 0;
  //   *activation_max = 6;
  // }
  // else if (activation == ir::Activation::RELU1)
  // {
  //   *activation_min = -1;
  //   *activation_max = 1;
  // }
  // else if (activation == ir::Activation::SIGMOID)
  // {
  //   *activation_min = 0;
  //   *activation_max = 1;
  // }
  // else if (activation == ir::Activation::NONE)
  // {
  //   *activation_min = std::numeric_limits<T>::lowest();
  //   *activation_max = std::numeric_limits<T>::max();
  // }
  else
  {
    throw std::runtime_error{"Unsupported fused activation function."};
  }
}

} // namespace util
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_UTIL_CALCULATE_ACTIVATION_RANGE_H__
