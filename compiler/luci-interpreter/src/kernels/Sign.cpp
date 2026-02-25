/*
 * Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Sign.h"

#include "kernels/Utils.h"

#include <cmath>

namespace luci_interpreter
{
namespace kernels
{

namespace
{

template <typename T>
inline void CalcSign(const T *input_data, const size_t num_elements, T *output_data)
{
  for (size_t i = 0; i < num_elements; ++i)
  {
    const T x = input_data[i];
    if constexpr (std::is_floating_point_v<T>)
    {
      if (std::isnan(x))
      {
        output_data[i] = x; // NaN -> NaN
        continue;
      }
    }
    output_data[i] = (T(0) < x) - (x < T(0)); // -1/0/1
  }
}

} // namespace

Sign::Sign(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Sign::configure()
{
  auto et = input()->element_type();
  LUCI_INTERPRETER_CHECK(et == DataType::FLOAT32 || et == DataType::FLOAT64 || et == DataType::S32);
  LUCI_INTERPRETER_CHECK(et == output()->element_type());
  output()->resize(input()->shape());
}

void Sign::execute() const
{
  switch (input()->element_type())
  {
    case DataType::S32:
      evalS32();
      break;
    case DataType::FLOAT32:
      evalFloat32();
      break;
    case DataType::FLOAT64:
      evalFloat64();
      break;
    default:
      throw std::runtime_error("luci-intp Sign Unsupported type.");
  }
}

template <typename T> void Sign::eval() const
{
  const int size = tflite::MatchingFlatSize(getTensorShape(input()), getTensorShape(output()));
  CalcSign(getTensorData<T>(input()), static_cast<size_t>(size), getTensorData<T>(output()));
}

void Sign::evalS32() const { eval<int32_t>(); }
void Sign::evalFloat32() const { eval<float>(); }
void Sign::evalFloat64() const { eval<double>(); }

} // namespace kernels
} // namespace luci_interpreter
