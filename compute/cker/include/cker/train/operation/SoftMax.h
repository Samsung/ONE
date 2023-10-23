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

#ifndef __NNFW_CKER_TRAIN_SOFTMAX_H__
#define __NNFW_CKER_TRAIN_SOFTMAX_H__

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"

namespace nnfw
{
namespace cker
{
namespace train
{

inline void SoftMaxGrad(const Shape &output_shape, const float *output_data,
                        const Shape &incoming_shape, const float *incoming_data,
                        const Shape &grad_shape, float *grad_data)
{
  // TODO Support 4dim softmax gradient
  assert(incoming_shape.DimensionsCount() == 2);
  MatchingFlatSize(output_shape, incoming_shape, grad_shape);

  const int batches = incoming_shape.Dims(0);
  const int width = incoming_shape.Dims(1);

  for (int b = 0; b < batches; ++b)
  {
    int b_offset = b * width;
    for (int w1 = 0; w1 < width; ++w1)
    {
      float sum = 0.0f;
      for (int w2 = 0; w2 < width; ++w2)
      {
        float val;
        if (w1 == w2)
        {
          val = output_data[b_offset + w2] * (1.f - output_data[b_offset + w2]);
        }
        else
        {
          val = -output_data[b_offset + w2] * output_data[b_offset + w1];
        }
        val *= incoming_data[b_offset + w2];
        sum += val;
      }
      grad_data[b_offset + w1] = sum;
    }
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_SOFTMAX_H__
