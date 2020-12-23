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

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Softmax::Softmax(const Tensor *input, Tensor *output, const SoftmaxParams &params)
    : KernelWithParams<SoftmaxParams>(params), _input(input), _output(output)
{
}

void Softmax::configure()
{
  assert(_input->element_type() == _output->element_type());
  _output->resize(_input->shape());
}

void Softmax::execute() const
{
  switch (_input->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void Softmax::evalFloat() const
{
  tflite::SoftmaxParams params{};
  params.beta = _params.beta;

  tflite::reference_ops::Softmax(params, getTensorShape(_input), getTensorData<float>(_input),
                                 getTensorShape(_output), getTensorData<float>(_output));
}

} // namespace kernels
} // namespace luci_interpreter
