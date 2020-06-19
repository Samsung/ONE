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

#include "kernels/LocalResponseNormalization.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

LocalResponseNormalization::LocalResponseNormalization(
    const Tensor *input, Tensor *output, const LocalResponseNormalizationParams &params)
    : KernelWithParams<LocalResponseNormalizationParams>(params), _input(input), _output(output)
{
}

void LocalResponseNormalization::configure()
{
  assert(_input->shape().num_dims() == 4);
  assert(_output->element_type() == DataType::FLOAT32);
  assert(_input->element_type() == _output->element_type());
  _output->resize(_input->shape());
}

void LocalResponseNormalization::execute() const
{
  switch (_output->element_type())
  {
    case DataType::FLOAT32:
      tflite::LocalResponseNormalizationParams op_params;
      op_params.range = params().radius;
      op_params.bias = params().bias;
      op_params.alpha = params().alpha;
      op_params.beta = params().beta;
      tflite::optimized_ops::LocalResponseNormalization(
          op_params, getTensorShape(_input), getTensorData<float>(_input), getTensorShape(_output),
          getTensorData<float>(_output));
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
