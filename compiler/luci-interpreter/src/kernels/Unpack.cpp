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

#include "kernels/Unpack.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Unpack::Unpack(const Tensor *input, std::vector<Tensor *> outputs, const UnpackParams &params)
  : KernelWithParams<UnpackParams>({input}, std::move(outputs), params)
{
}

void Unpack::configure()
{
  const Shape &input_shape = input()->shape();

  int axis = _params.axis;
  if (axis < 0)
    axis += input()->shape().num_dims();
  assert(axis >= 0 && axis < input_shape.num_dims());

  Shape output_shape(input_shape.num_dims() - 1);
  int out_index = 0;
  for (int in_index = 0; in_index < input_shape.num_dims(); ++in_index)
  {
    if (in_index != axis)
      output_shape.dim(out_index++) = input_shape.dim(in_index);
  }

  for (Tensor *output : _outputs)
  {
    assert(output->element_type() == input()->element_type());
    output->resize(output_shape);
  }
}

template <typename T> void Unpack::executeImpl() const
{
  tflite::UnpackParams params{};
  params.axis = _params.axis;
  params.num_split = _outputs.size();
  VectorOfTensors<T, false> all_outputs(_outputs);
  tflite::reference_ops::Unpack<T>(params, getTensorShape(input()), getTensorData<T>(input()),
                                   **all_outputs.shapes(), all_outputs.data());
}

void Unpack::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      return executeImpl<float>();
    case DataType::U8:
      return executeImpl<uint8_t>();
    default:
      throw std::runtime_error("luci-intp Unpack Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
