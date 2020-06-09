/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_MATRIX_BAND_PART_H__
#define __NNFW_CKER_MATRIX_BAND_PART_H__

#include "cker/Shape.h"

#include <algorithm>

namespace nnfw
{
namespace cker
{

void MatrixBandPart(int num_lower_diags, int num_upper_diags, const Shape &input_shape,
                    const float *input_data, const Shape &output_shape, float *output_data)
{
  auto last_dim = input_shape.DimensionsCount() - 1;

  int batch_num = 0;
  for (int dim = 0; dim < last_dim - 2; dim++)
  {
    batch_num += input_shape.Dims(dim);
  }

  const int row_num = input_shape.Dims(last_dim - 1);
  const int col_num = input_shape.Dims(last_dim);

  std::fill(output_data, output_data + output_shape.FlatSize(), 0); // output matrix init

  // reference code, without multithreading
  for (int batch = 0; batch < batch_num; ++batch)
  {
    for (int row = 0; row < row_num; ++row)
    {
      auto output = output_data + (batch * row_num * col_num + row * col_num);
      auto input = input_data + (batch * row_num * col_num + row * col_num);

      const int band_start =
          num_lower_diags < 0 ? 0 : std::min(col_num, std::max(int{0}, row - num_lower_diags));
      const int band_end = num_upper_diags < 0 ? col_num : std::min(static_cast<int>(col_num),
                                                                    row + num_upper_diags + 1);

      for (int band_idx = band_start; band_idx < band_end; band_idx++)
      {
        output[band_idx] = input[band_idx];
      }
    }
  }
}
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_MATRIX_BAND_PART_H__
