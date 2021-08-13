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

#include "kernels/Floor.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/floor.h>

namespace luci_interpreter
{

namespace kernels
{

Floor::Floor(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Floor::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  output()->resize(input()->shape());
}

void Floor::execute() const
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

void Floor::evalFloat() const
{
  tflite::reference_ops::Floor(getTensorShape(input()), getTensorData<float>(input()),
                               getTensorShape(output()), getTensorData<float>(output()));
}

void Floor::evalQ16() const
{
  auto input_shape = getTensorShape(input());
  auto output_shape = getTensorShape(output());

  std::vector<float> input_data(input_shape.FlatSize());
  std::vector<float> output_data(output_shape.FlatSize());

  float input_scale = input()->scale();
  int32_t input_zpoint = input()->zero_point();

  float output_scale = output()->scale();
  int32_t output_zpoint = output()->zero_point();

  for (int i = 0; i < input_shape.FlatSize(); ++i)
    input_data[i] = (input()->data<int16_t>()[i] - input_zpoint) * input_scale;

  tflite::reference_ops::Floor(input_shape, input_data.data(),
                               output_shape, output_data.data());

  for (int i = 0; i < input_shape.FlatSize(); ++i)
    output()->data<int16_t>()[i] = std::round(output_data[i]/output_scale) + output_zpoint;
}

} // namespace kernels
} // namespace luci_interpreter
