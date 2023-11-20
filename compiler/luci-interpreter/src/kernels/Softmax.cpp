/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Softmax.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/softmax.h>
#include "PALSoftmax.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Softmax::Softmax(const Tensor *input, Tensor *output, const SoftmaxParams &params)
  : KernelWithParams<SoftmaxParams>({input}, {output}, params)
{
}

void Softmax::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() >= 1);
  if (input()->element_type() == DataType::U8 || input()->element_type() == DataType::S8)
  {
    LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::S8 || output()->zero_point() == 0);
    LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::U8 ||
                           output()->zero_point() == std::numeric_limits<int8_t>::min());
    tflite::SoftmaxParams op_params{};
    op_params.table = _table;
    luci_interpreter_pal::PopulateSoftmaxLookupTable(&op_params, input()->scale(), params().beta);
  }
  output()->resize(input()->shape());
}

void Softmax::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::S8:
      evalQuantized<int8_t>();
      break;
    case DataType::U8:
      evalQuantized<uint8_t>();
      break;
    default:
      throw std::runtime_error("luci-intp Softmax Unsupported type.");
  }
}

void Softmax::evalFloat() const
{
  tflite::SoftmaxParams op_params{};
  op_params.beta = params().beta;

  tflite::reference_ops::Softmax(op_params, getTensorShape(input()), getTensorData<float>(input()),
                                 getTensorShape(output()), getTensorData<float>(output()));
}

template <typename T> void Softmax::evalQuantized() const
{
  tflite::SoftmaxParams op_params{};
  op_params.table = const_cast<float *>(_table);
  op_params.zero_point = output()->zero_point();
  op_params.scale = output()->scale();
  luci_interpreter_pal::InitializeParams(&op_params, input()->scale(), params().beta);
  luci_interpreter_pal::Softmax(op_params, getTensorShape(input()), getTensorData<T>(input()),
                                getTensorShape(output()), getTensorData<T>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
