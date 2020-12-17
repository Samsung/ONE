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

#include "CappedReLU.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

#include "Common.h"

#include <algorithm>
#include <cstdint>

namespace mir_interpreter
{

template <typename T> struct CappedReLUImpl
{
  static void run(const mir::TensorVariant &arg, float cap, mir::TensorVariant &result);
};

template <typename T>
void CappedReLUImpl<T>::run(const mir::TensorVariant &arg, float cap, mir::TensorVariant &result)
{
  mir::Tensor<T> arg_accessor(arg);
  mir::Tensor<T> res_accessor(result);

  for (const auto &index : mir::ShapeRange(result.getShape()))
  {
    res_accessor.at(index) = std::min(std::max(arg_accessor.at(index), T(0)), static_cast<T>(cap));
  }
}

static float dequantize(uint8_t x, const mir::AffineQuantization &q)
{
  return (static_cast<int>(x) - q.getZeroPoint()) * q.getScale();
}

static uint8_t quantize(float x, const mir::AffineQuantization &q)
{
  return (static_cast<float>(x) / q.getScale() + q.getZeroPoint());
}

template <> struct CappedReLUImpl<uint8_t>
{
  static void run(const mir::TensorVariant &arg, float cap, mir::TensorVariant &result)
  {
    mir::Tensor<uint8_t> arg_accessor(arg);
    mir::Tensor<uint8_t> res_accessor(result);

    auto quant_info = arg.getType().getQuantization();
    assert(!quant_info.empty());

    for (const auto &index : mir::ShapeRange(result.getShape()))
    {
      auto value = dequantize(arg_accessor.at(index), quant_info);
      auto out_value =
        quantize(std::min(std::max(value, 0.0f), cap), result.getType().getQuantization());
      res_accessor.at(index) = out_value;
    }
  }
};

void CappedReLU(const mir::TensorVariant &arg, float cap, mir::TensorVariant &result)
{
  dispatch<CappedReLUImpl>(arg.getElementType(), arg, cap, result);
}

} // namespace mir_interpreter
