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

#include "kernels/Pow.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

Pow::Pow(const Tensor *input1, const Tensor *input2, Tensor *output)
    : Kernel({input1, input2}, {output})
{
}

void Pow::configure()
{
  if (input1()->shape() != input2()->shape())
  {
    throw std::runtime_error("Input Tensor Shape Mismatch.");
  }

  if (input1()->element_type() != input2()->element_type())
  {
    throw std::runtime_error("Input Tensor Data Type Mismatch.");
  }

  const Shape &output_shape = input1()->shape();

  output()->resize(output_shape);
}

void Pow::execute() const
{
  tflite::reference_ops::Pow(getTensorShape(input1()), getTensorData<float>(input1()),
                             getTensorShape(input2()), getTensorData<float>(input2()),
                             getTensorShape(output()), getTensorData<float>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
