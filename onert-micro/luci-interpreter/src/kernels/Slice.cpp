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

#include "kernels/Slice.h"
#include "Utils.h"
#include "PALSlice.h"

#include <cassert>
#include <cstring>

namespace luci_interpreter
{

namespace kernels
{
const int max_dim = 4;

Slice::Slice(const Tensor *input, const Tensor *begin, const Tensor *size, Tensor *output)
  : Kernel({input, begin, size}, {output})
{
}

template <typename T>
Shape calculateOutputShape(const Tensor *input, const Tensor *begin, const Tensor *size)
{
  Shape output_shape = Shape(input->shape().num_dims());
  for (int idx = 0; idx < input->shape().num_dims(); idx++)
  {
    T size_value = getTensorData<T>(size)[idx];
    if (size_value < 0)
    {
      if (size_value != -1)
      {
        assert(false && "Invalid size.");
      }
      size_value = input->shape().dim(idx) - getTensorData<T>(begin)[idx];
    }
    else
    {
      if (input->shape().dim(idx) < getTensorData<T>(begin)[idx] + size_value)
      {
        assert(false && "Invalid begin and size.");
      }
    }
    output_shape.dim(idx) = static_cast<int>(size_value);
  }
  return output_shape;
}

template <typename T>
void getBeginAndSizeVectors(int dimensions, const Tensor *begin, const Tensor *size,
                            std::vector<int> *begins, std::vector<int> *sizes)
{
  for (int idx = dimensions - 1; idx >= 0; --idx)
  {
    begins->push_back(getTensorData<T>(begin)[idx]);
    sizes->push_back(getTensorData<T>(size)[idx]);
  }
}

void Slice::configure()
{
  assert(input()->element_type() == output()->element_type());
  assert(begin()->element_type() == DataType::S32 || begin()->element_type() == DataType::S64);
  assert(size()->element_type() == DataType::S32 || size()->element_type() == DataType::S64);
  assert(begin()->shape().num_dims() == 1);
  assert(size()->shape().num_dims() == 1);
  assert(input()->shape().num_dims() <= max_dim);
  // TODO: enable it only if kernel with dynamic shapes
  if (begin()->element_type() == DataType::S32)
  {
    output()->resize(calculateOutputShape<int32_t>(input(), begin(), size()));
  }
  else if (begin()->element_type() == DataType::S64)
  {
    output()->resize(calculateOutputShape<int64_t>(input(), begin(), size()));
  }
  else
  {
    assert(false && "Unsupported type.");
  }
}

void Slice::execute() const
{
  std::vector<int> begins;
  begins.reserve(max_dim);
  std::vector<int> sizes;
  sizes.reserve(max_dim);
  if (begin()->element_type() == DataType::S32)
  {
    getBeginAndSizeVectors<int32_t>(input()->shape().num_dims(), begin(), size(), &begins, &sizes);
  }
  else if (begin()->element_type() == DataType::S64)
  {
    getBeginAndSizeVectors<int64_t>(input()->shape().num_dims(), begin(), size(), &begins, &sizes);
  }
  else
  {
    assert(false && "Unsupported begin type.");
  }
  for (int i = input()->shape().num_dims(); i < max_dim; ++i)
  {
    begins.push_back(0);
    sizes.push_back(1);
  }

  assert(begins.size() == 4);
  assert(sizes.size() == 4);
  tflite::SliceParams op_params{};
  op_params.begin_count = 4;
  op_params.size_count = 4;
  for (int i = 0; i < 4; i++)
  {
    op_params.begin[i] = begins[3 - i];
    op_params.size[i] = sizes[3 - i];
  }
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      luci_interpreter_pal::Slice(op_params, getTensorShape(input()), getTensorData<float>(input()),
                                  getTensorShape(output()), getTensorData<float>(output()));
      break;
    case DataType::U8:
      luci_interpreter_pal::Slice(op_params, getTensorShape(input()),
                                  getTensorData<uint8_t>(input()), getTensorShape(output()),
                                  getTensorData<uint8_t>(output()));
      break;
    case DataType::S8:
      luci_interpreter_pal::Slice(op_params, getTensorShape(input()),
                                  getTensorData<int8_t>(input()), getTensorShape(output()),
                                  getTensorData<int8_t>(output()));
      break;
    default:
      assert(false && "Unsupported input type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
