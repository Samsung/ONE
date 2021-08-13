/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Neg.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Neg::Neg(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Neg::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());

  output()->resize(input()->shape());
}

void Neg::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::S16:
      evalQ16();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void Neg::evalFloat() const
{
  tflite::reference_ops::Negate(getTensorShape(input()), getTensorData<float>(input()),
                                getTensorShape(output()), getTensorData<float>(output()));
}

void Neg::evalQ16() const
{
  auto input_shape = getTensorShape(input());

  for (int i = 0; i < input_shape.FlatSize(); ++i)
    output()->data<int16_t>()[i] = -input()->data<int16_t>()[i];
}

} // namespace kernels
} // namespace luci_interpreter
