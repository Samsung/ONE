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

#include "kernels/Transpose.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/transpose.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Transpose::Transpose(const Tensor *input, const Tensor *perm, Tensor *output)
  : Kernel({input, perm}, {output})
{
}

void Transpose::configure()
{
  // Transpose op only supports 1D-4D input arrays.
  int dims = input()->shape().num_dims();
  const int32_t *perm_data = getTensorData<int32_t>(perm());

  assert(input()->shape().num_dims() <= 4);
  assert(input()->element_type() == output()->element_type());

  assert(perm()->shape().num_dims() == 1);
  assert(perm()->shape().dim(0) == dims);

  Shape output_shape(dims);
  for (int i = 0; i < dims; i++)
  {
    assert(perm_data[i] < dims && perm_data[i] >= 0);
    output_shape.dim(i) = input()->shape().dim(perm_data[i]);
  }

  output()->resize(output_shape);
}

void Transpose::execute() const
{
  tflite::TransposeParams params{};
  const int32_t *perm_data = getTensorData<int32_t>(perm());
  const int32_t size = perm()->shape().dim(0);
  params.perm_count = size;
  for (int i = 0; i < size; i++)
    params.perm[i] = perm_data[i];
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      tflite::reference_ops::Transpose(params, getTensorShape(input()),
                                       getTensorData<float>(input()), getTensorShape(output()),
                                       getTensorData<float>(output()));
      break;
    case DataType::U8:
      tflite::reference_ops::Transpose(params, getTensorShape(input()),
                                       getTensorData<uint8_t>(input()), getTensorShape(output()),
                                       getTensorData<uint8_t>(output()));
      break;
    default:
      throw std::runtime_error("luci-intp Transpose Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
