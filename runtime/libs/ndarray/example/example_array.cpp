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

#include "ndarray/Array.h"

#include <iostream>
#include <iterator>

using namespace ndarray;

void gather_array(const Array<float> &input, Array<float> &output, const Array<int> &indices)
{
  assert(indices.shape().rank() == 3);
  assert(input.shape().rank() == 3);
  assert(indices.shape().dim(1) == input.shape().rank());

  for (size_t i = 0; i < indices.shape().dim(0); ++i)
  {
    for (size_t j = 0; j < indices.shape().dim(1); ++j)
    {
      auto index = indices.slice(i, j);
      output.slice(i, j).assign(input.slice(index[0], index[1]));
    }
  }
}

int main()
{
  // fill tensor of shape[3,3,4] with sequential numbers from [0..36)
  Shape in_shape{3, 3, 4};
  std::vector<float> input_data(in_shape.element_count());
  for (size_t i = 0; i < in_shape.element_count(); ++i)
    input_data[i] = i;

  Array<float> input(input_data.data(), in_shape);

  // select column-vectors on main diagonal
  Shape indices_shape{1, 3, 2};
  std::vector<int> indices_data(indices_shape.element_count());
  Array<int> indices(indices_data.data(), indices_shape);

  indices.slice(0, 0) = {0, 0};
  indices.slice(0, 1) = {1, 1};
  indices.slice(0, 2) = {2, 2};

  Shape output_shape{1, 3, 4};
  std::vector<float> output_data(output_shape.element_count());

  Array<float> output(output_data.data(), output_shape);

  gather_array(input, output, indices);

  for (size_t i = 0; i < indices_shape.dim(0); ++i)
  {
    for (size_t j = 0; j < indices_shape.dim(1); ++j)
    {
      auto output_piece = output.slice(i, j);
      std::ostream_iterator<int> cout_it(std::cout, ", ");
      std::copy(output_piece.begin(), output_piece.end(), cout_it);
      std::cout << std::endl;
    }
  }
}
