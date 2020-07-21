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

#include "Split.h"

#include "Utils.h"

#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

namespace luci_interpreter
{
namespace kernels
{

Split::Split(const Tensor *axis, const Tensor *input, std::vector<Tensor *> outputs)
    : _axis(axis), _input(input), _outputs(std::move(outputs))
{
}

void Split::configure()
{
  assert(_axis->shape().num_elements() == 1);
  _axis_value = getTensorData<int32_t>(_axis)[0];
  if (_axis_value < 0)
    _axis_value += _input->shape().num_dims();
  assert(_axis_value >= 0 && _axis_value < _input->shape().num_dims());

  const int32_t input_size = _input->shape().dim(_axis_value);
  assert(input_size % _outputs.size() == 0);
  const int32_t slice_size = input_size / _outputs.size();

  Shape output_shape = _input->shape();
  output_shape.dim(_axis_value) = slice_size;
  for (Tensor *_output : _outputs)
  {
    _output->resize(output_shape);
  }
}

void Split::execute() const
{
  tflite::SplitParams params{};
  params.num_split = _outputs.size();
  params.axis = _axis_value;

#define TF_LITE_SPLIT(scalar)                                                                   \
  {                                                                                             \
    VectorOfTensors<scalar, false> all_outputs(_outputs);                                       \
    tflite::optimized_ops::Split(params, getTensorShape(_input), getTensorData<scalar>(_input), \
                                 all_outputs.shapes(), all_outputs.data());                     \
  }

  switch (_input->element_type())
  {
    case DataType::FLOAT32:
      TF_LITE_SPLIT(float);
      break;
    case DataType::U8:
      TF_LITE_SPLIT(uint8_t);
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
#undef TF_LITE_SPLIT
}

} // namespace kernels
} // namespace luci_interpreter
