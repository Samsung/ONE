/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/FloorMod.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/binary_function.h>
#include <cmath>

namespace
{

template <typename T> T FloorDivFunc(T input1, T input2)
{
  struct FloatMod
  {
    float operator()(const float lhs, const float rhs) const { return std::fmod(lhs, rhs); }
  };
  using ModFunc =
    typename std::conditional<std::is_integral<T>::value, std::modulus<T>, FloatMod>::type;
  ModFunc mod_func;
  T trunc_mod = mod_func(input1, input2);
  return (trunc_mod != 0) && ((input2 < 0) != (trunc_mod < 0)) ? (trunc_mod + input2) : trunc_mod;
}

} // namespace

namespace luci_interpreter
{

namespace kernels
{

FloorMod::FloorMod(const Tensor *x, const Tensor *y, Tensor *output) : Kernel({x, y}, {output}) {}

void FloorMod::configure()
{
  LUCI_INTERPRETER_CHECK(x()->element_type() == output()->element_type());
  LUCI_INTERPRETER_CHECK(y()->element_type() == output()->element_type());

  output()->resize(calculateShapeForBroadcast(x()->shape(), y()->shape()));
}

void FloorMod::execute() const
{
  switch (x()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::S8:
      evalInteger<int8_t>();
      break;
    case DataType::S16:
      evalInteger<int16_t>();
      break;
    case DataType::S32:
      evalInteger<int32_t>();
      break;
    case DataType::S64:
      evalInteger<int64_t>();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void FloorMod::evalFloat() const
{
  const auto x_data = getTensorData<float>(x());
  const auto y_data = getTensorData<float>(y());

  for (int i = 0; i < getTensorShape(y()).FlatSize(); ++i)
  {
    LUCI_INTERPRETER_CHECK(y_data[i] != 0);
  }

  if (x()->shape() != y()->shape())
  {
    tflite::reference_ops::BroadcastBinaryFunction4DSlow<float, float, float>(
      getTensorShape(x()), x_data, getTensorShape(y()), y_data, getTensorShape(output()),
      getTensorData<float>(output()), FloorDivFunc<float>);
  }
  else
  {
    tflite::reference_ops::BinaryFunction<float, float, float>(
      getTensorShape(x()), x_data, getTensorShape(y()), y_data, getTensorShape(output()),
      getTensorData<float>(output()), FloorDivFunc<float>);
  }
}

template <typename T> void FloorMod::evalInteger() const
{
  const auto x_data = getTensorData<T>(x());
  const auto y_data = getTensorData<T>(y());

  // Check the denominator
  const auto y_data_type = y()->element_type();
  if (y_data_type == DataType::S8 || y_data_type == DataType::S16 || y_data_type == DataType::S32 ||
      y_data_type == DataType::S64)
  {
    for (int i = 0; i < getTensorShape(y()).FlatSize(); ++i)
    {
      LUCI_INTERPRETER_CHECK(y_data[i] != 0);
    }
  }

  if (x()->shape() != y()->shape())
  {
    tflite::reference_ops::BroadcastBinaryFunction4DSlow<T, T, T>(
      getTensorShape(x()), x_data, getTensorShape(y()), y_data, getTensorShape(output()),
      getTensorData<T>(output()), FloorDivFunc<T>);
  }
  else
  {
    tflite::reference_ops::BinaryFunction<T, T, T>(getTensorShape(x()), x_data, getTensorShape(y()),
                                                   y_data, getTensorShape(output()),
                                                   getTensorData<T>(output()), FloorDivFunc<T>);
  }
}

} // namespace kernels
} // namespace luci_interpreter
