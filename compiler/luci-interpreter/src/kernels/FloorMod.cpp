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
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void FloorMod::evalFloat() const
{
  auto FloorModFunc = [](float x, float y) {
    float trunc_mod = std::fmod(x, y);

    return (trunc_mod != 0) && ((y < 0) != (trunc_mod < 0)) ? (trunc_mod + y) : trunc_mod;
  };

  const auto x_data = getTensorData<float>(x());
  const auto y_data = getTensorData<float>(y());

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
    tflite::reference_ops::BroadcastBinaryFunction4DSlow<float, float, float>(
      getTensorShape(x()), x_data, getTensorShape(y()), y_data, getTensorShape(output()),
      getTensorData<float>(output()), FloorModFunc);
  }
  else
  {
    tflite::reference_ops::BinaryFunction<float, float, float>(
      getTensorShape(x()), x_data, getTensorShape(y()), y_data, getTensorShape(output()),
      getTensorData<float>(output()), FloorModFunc);
  }
}

} // namespace kernels
} // namespace luci_interpreter
