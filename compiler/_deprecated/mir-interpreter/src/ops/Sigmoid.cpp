/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Sigmoid.h"
#include "Common.h"

#include <mir/ShapeRange.h>
#include <mir/Tensor.h>

#include <cmath>

namespace mir_interpreter
{

template <typename T> struct SigmoidImpl
{
  static void run(const mir::TensorVariant &arg, mir::TensorVariant &result);
};

template <typename T>
void SigmoidImpl<T>::run(const mir::TensorVariant &arg, mir::TensorVariant &result)
{
  mir::Tensor<T> arg_accessor(arg);
  mir::Tensor<T> res_accessor(result);

  for (const auto &index : mir::ShapeRange(result.getShape()))
  {
    res_accessor.at(index) = 1.0f / (1.0f + std::exp(-arg_accessor.at(index)));
  }
}

template <> struct SigmoidImpl<uint8_t>
{
  static void run(const mir::TensorVariant &arg, mir::TensorVariant &result)
  {
    throw std::runtime_error{"NYI"};
  }
};

void Sigmoid(const mir::TensorVariant &arg, mir::TensorVariant &result)
{
  dispatch<SigmoidImpl>(arg.getElementType(), arg, result);
};

} // namespace mir_interpreter
