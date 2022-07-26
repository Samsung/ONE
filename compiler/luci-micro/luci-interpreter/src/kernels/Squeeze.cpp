/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Squeeze.h"

#include "kernels/Utils.h"

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

Squeeze::Squeeze(const Tensor *input, Tensor *output, const SqueezeParams &params)
  : KernelWithParams<SqueezeParams>({input}, {output}, params)
{
}

void Squeeze::configure()
{
  int input_num_dims = input()->shape().num_dims();
  int num_squeeze_dims = params().squeeze_dims.size();
  assert(input_num_dims <= 8);
  bool should_squeeze[8] = {false};
  int num_squeezed_dims = 0;
  if (num_squeeze_dims == 0)
  {
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      if (input()->shape().dim(idx) == 1)
      {
        should_squeeze[idx] = true;
        ++num_squeezed_dims;
      }
    }
  }
  else
  {
    for (int idx = 0; idx < num_squeeze_dims; ++idx)
    {
      int current = params().squeeze_dims[idx] < 0 ? params().squeeze_dims[idx] + input_num_dims
                                                   : params().squeeze_dims[idx];
      assert(current >= 0 && current < input_num_dims && input()->shape().dim(current) == 1);
      if (!should_squeeze[current])
        ++num_squeezed_dims;
      should_squeeze[current] = true;
    }
  }
  Shape output_shape(input_num_dims - num_squeezed_dims);
  for (int in_idx = 0, out_idx = 0; in_idx < input_num_dims; ++in_idx)
  {
    if (!should_squeeze[in_idx])
    {
      output_shape.dim(out_idx++) = input()->shape().dim(in_idx);
    }
  }
  output()->resize(output_shape);
}

void Squeeze::execute() const
{
  assert(input()->shape().num_elements() == output()->shape().num_elements());

  const auto *input_data = input()->data<void>();
  auto *output_data = output()->data<void>();
  std::memcpy(output_data, input_data,
              getDataTypeSize(input()->element_type()) * input()->shape().num_elements());
}

} // namespace kernels
} // namespace luci_interpreter
