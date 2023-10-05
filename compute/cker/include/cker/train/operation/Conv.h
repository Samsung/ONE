/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_OPERATION_CONV_H__
#define __NNFW_CKER_TRAIN_OPERATION_CONV_H__

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"

#include <Eigen/Core>

namespace nnfw
{
namespace cker
{
namespace train
{

template <typename T>
inline void ConvBiasGrad(const Shape &incomming_shape, const T *incomming_data,
                         const Shape &output_shape, T *output_data)
{
  assert(incomming_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 1);
  assert(incomming_shape.Dims(3) == output_shape.Dims(0));
  const auto nums_batches = incomming_shape.Dims(0);
  const auto nums_heights = incomming_shape.Dims(1);
  const auto nums_widths = incomming_shape.Dims(2);
  const auto nums_channels = incomming_shape.Dims(3);

  memset(output_data, 0, incomming_shape.FlatSize() * sizeof(T));

  for (int b = 0; b < nums_batches; ++b)
  {
    for (int h = 0; h < nums_heights; ++h)
    {
      for (int w = 0; w < nums_widths; ++h)
      {
        for (int c = 0; c < nums_channels; ++c)
        {
          output_data[c] += incomming_data[Offset(incomming_shape, b, h, w, c)];
        }
      }
    }
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_RELU_H__
