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

#include "kernels/LogicalOr.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

namespace luci_interpreter
{
namespace kernels
{

LogicalOr::LogicalOr(const Tensor *input1, const Tensor *input2, Tensor *output)
    : Kernel({input1, input2}, {output})
{
}

void LogicalOr::configure()
{
  LUCI_INTERPRETER_CHECK(input1()->element_type() == input2()->element_type());
  LUCI_INTERPRETER_CHECK(input1()->element_type() == DataType::BOOL);
  output()->resize(calculateShapeForBroadcast(input1()->shape(), input2()->shape()));
}

void LogicalOr::execute() const
{
  auto func = [](bool x, bool y) { return x || y; };
  if (haveSameShape(input1(), input2()))
  {
    tflite::reference_ops::BinaryFunction<bool, bool, bool>(
        getTensorShape(input1()), getTensorData<bool>(input1()), getTensorShape(input2()),
        getTensorData<bool>(input2()), getTensorShape(output()), getTensorData<bool>(output()),
        func);
  }
  else
  {
    tflite::reference_ops::BroadcastBinaryFunction4DSlow<bool, bool, bool>(
        getTensorShape(input1()), getTensorData<bool>(input1()), getTensorShape(input2()),
        getTensorData<bool>(input2()), getTensorShape(output()), getTensorData<bool>(output()),
        func);
  }
}

} // namespace kernels
} // namespace luci_interpreter
