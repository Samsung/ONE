/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <array>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>

void gather_no_array(const float *in_data, const std::array<size_t, 3> &dims, float *out_data,
                     const std::array<size_t, 3> &out_dims, //[nselections,
                     const int *indices, const std::array<size_t, 3> &indices_dims)
{
  assert(indices_dims[1] == dims.size());

  for (int i = 0; i < indices_dims[0]; ++i)
  {
    for (int j = 0; j < indices_dims[1]; ++j)
    {
      const int *index_ptr = indices + i * indices_dims[2] * indices_dims[1] + j * indices_dims[2];

      size_t in_offset = index_ptr[0] * dims[2] * dims[1] + index_ptr[1] * dims[2];

      const float *in_ptr = in_data + in_offset;

      size_t out_offset = i * out_dims[2] * out_dims[1] + j * out_dims[2];

      float *out_ptr = out_data + out_offset;

      for (int k = 0; k < dims[2]; ++k)
      {
        out_ptr[k] = in_ptr[k];
      }
    }
  }
}

int main()
{
  std::array<size_t, 3> in_dims{3, 3, 4};
  std::vector<float> input(3 * 3 * 4);
  for (size_t i = 0; i < 3 * 3 * 4; ++i)
    input[i] = i;

  std::array<size_t, 3> indices_shape{1, 3, 2};
  std::vector<int> indices(1 * 3 * 2);

  indices[0] = 0;
  indices[1] = 0;
  indices[2] = 1;
  indices[3] = 1;
  indices[4] = 2;
  indices[5] = 2;

  std::array<size_t, 3> output_dims{1, 3, 4};
  std::vector<float> output(1 * 3 * 4);

  gather_no_array(input.data(), in_dims, output.data(), output_dims, indices.data(), indices_shape);

  for (size_t i = 0; i < output_dims[0]; ++i)
  {
    for (size_t j = 0; j < output_dims[1]; ++j)
    {
      auto out_ptr = output.data() + i * output_dims[1] * output_dims[2] + j * output_dims[2];
      for (size_t k = 0; k < output_dims[2]; ++k)
      {
        std::cout << out_ptr[k] << ", ";
      }
      std::cout << std::endl;
    }
  }
}
