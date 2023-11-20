/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/SquaredDifference.h"

#include "kernels/Utils.h"

#include "kernels/BinaryOpCommon.h"

namespace luci_interpreter
{
namespace kernels
{

SquaredDifference::SquaredDifference(const Tensor *input1, const Tensor *input2, Tensor *output)
  : Kernel({input1, input2}, {output})
{
}

void SquaredDifference::configure()
{
  LUCI_INTERPRETER_CHECK(input1()->element_type() == input2()->element_type())
  LUCI_INTERPRETER_CHECK(input1()->element_type() == output()->element_type())
  output()->resize(calculateShapeForBroadcast(input1()->shape(), input2()->shape()));
}

void SquaredDifference::execute() const
{
  switch (input1()->element_type())
  {
    case DataType::FLOAT32:
      evalSquaredDifference<float>();
      break;
    default:
      throw std::runtime_error("luci-intp SquaredDifference Unsupported type.");
  }
}

template <typename T> inline void SquaredDifference::evalSquaredDifference() const
{
  BinaryOpBroadcastSlow(getTensorShape(input1()), getTensorData<T>(input1()),
                        getTensorShape(input2()), getTensorData<T>(input2()),
                        getTensorShape(output()), getTensorData<T>(output()), [](T x, T y) {
                          const T difference = x - y;
                          return difference * difference;
                        });
}

} // namespace kernels
} // namespace luci_interpreter
