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

#include "ReLU.h"
#include "Common.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

#include <cmath>

namespace mir_interpreter
{

template <typename T> struct ELUImpl
{
  static void run(const mir::TensorVariant &arg, float alpha, mir::TensorVariant &result);
};

template <typename T>
void ELUImpl<T>::run(const mir::TensorVariant &arg, float alpha, mir::TensorVariant &result)
{
  mir::Tensor<T> arg_accessor(arg);
  mir::Tensor<T> res_accessor(result);

  for (const auto &index : mir::ShapeRange(result.getShape()))
  {
    const T x = arg_accessor.at(index);
    res_accessor.at(index) = x < 0 ? alpha * (std::exp(x) - 1) : x;
  }
}

void ELU(const mir::TensorVariant &arg, float alpha, mir::TensorVariant &result)
{
  dispatch<ELUImpl>(result.getElementType(), arg, alpha, result);
}

} // namespace mir_interpreter
