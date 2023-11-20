/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/FloorDiv.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/binary_function.h>
#include <cmath>

namespace luci_interpreter
{

namespace kernels
{

FloorDiv::FloorDiv(const Tensor *input, const Tensor *alpha, Tensor *output)
  : Kernel({input, alpha}, {output})
{
}

void FloorDiv::configure()
{
  LUCI_INTERPRETER_CHECK(x()->element_type() == output()->element_type());
  LUCI_INTERPRETER_CHECK(y()->element_type() == output()->element_type());

  output()->resize(calculateShapeForBroadcast(x()->shape(), y()->shape()));
}

void FloorDiv::execute() const
{
  switch (x()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("luci-intp FloorDiv Unsupported type.");
  }
}

void FloorDiv::evalFloat() const
{
  auto FloorDivFunc = [](float x, float y) -> float {
    return std::floor(static_cast<double>(x) / static_cast<double>(y));
  };

  const auto x_data = getTensorData<float>(x());
  const auto y_data = getTensorData<float>(y());

  // Check the denominator
  for (int i = 0; i < getTensorShape(y()).FlatSize(); ++i)
  {
    LUCI_INTERPRETER_CHECK(y_data[i] != 0);
  }

  if (x()->shape() != y()->shape())
  {
    tflite::reference_ops::BroadcastBinaryFunction4DSlow<float, float, float>(
      getTensorShape(x()), x_data, getTensorShape(y()), y_data, getTensorShape(output()),
      getTensorData<float>(output()), FloorDivFunc);
  }
  else
  {
    tflite::reference_ops::BinaryFunction<float, float, float>(
      getTensorShape(x()), x_data, getTensorShape(y()), y_data, getTensorShape(output()),
      getTensorData<float>(output()), FloorDivFunc);
  }
}

} // namespace kernels
} // namespace luci_interpreter
