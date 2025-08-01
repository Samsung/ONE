/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Round.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/round.h>

namespace luci_interpreter
{

namespace kernels
{

Round::Round(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Round::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  output()->resize(input()->shape());
}

void Round::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;

    default:
      throw std::runtime_error("luci-intp Round Unsupported type.");
  }
}

void Round::evalFloat() const
{
  tflite::reference_ops::Round(getTensorShape(input()), getTensorData<float>(input()),
                               getTensorShape(output()), getTensorData<float>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
