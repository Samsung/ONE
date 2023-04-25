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

#include "kernels/LogSoftmax.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/log_softmax.h>

#include "PALLogSoftmax.h"

namespace luci_interpreter
{
namespace kernels
{

LogSoftmax::LogSoftmax(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void LogSoftmax::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  if (input()->element_type() == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(output()->scale() == 16. / 256);
    LUCI_INTERPRETER_CHECK(output()->zero_point() == 255);

    tflite::SoftmaxParams params{};

    params.table = _table;
    params.beta = 1.0;
    luci_interpreter_pal::PopulateSoftmaxLookupTable(&params, input()->scale(), params.beta);
  }
  // TODO: enable it only if kernel with dynamic shapes
  output()->resize(input()->shape());
}

void LogSoftmax::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    default:
      assert(false && "Unsupported type.");
  }
}

void LogSoftmax::evalFloat() const
{
  tflite::SoftmaxParams params{};
  tflite::reference_ops::LogSoftmax(params, getTensorShape(input()), getTensorData<float>(input()),
                                    getTensorShape(output()), getTensorData<float>(output()));
}

void LogSoftmax::evalQuantized() const
{
  const auto input_shape = getTensorShape(input());
  const auto output_shape = getTensorShape(output());
  const auto input_scale = input()->scale();
  uint8_t *output_data = getTensorData<uint8_t>(output());
  const uint8_t *input_data = getTensorData<uint8_t>(input());
  const float beta = 1.0;

  tflite::SoftmaxParams params{};

  params.table = const_cast<float *>(_table);
  params.zero_point = output()->zero_point();
  params.scale = output()->scale();

  luci_interpreter_pal::InitializeParams(&params, input_scale, beta);
  luci_interpreter_pal::LogSoftmax(params, input_scale, input_shape, input_data, output_shape,
                                   output_data);
}

} // namespace kernels
} // namespace luci_interpreter
