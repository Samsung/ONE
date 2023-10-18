/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_SOFTMAX_H__
#define __NNFW_CKER_TRAIN_SOFTMAX_H__

#include <iostream>
#include "cker/Shape.h"
// #include "cker/Utils.h"
// #include "cker/Types.h"
#include "cker/eigen/Utils.h"

// #if __aarch64__ && __clang__
// #define TFLITE_SOFTMAX_USE_UINT16_LUT
// #endif

// #include <Eigen/Core>
// #include <fixedpoint/fixedpoint.h>
// #include <cmath>

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
  assert(incoming_shape.DimensionsCount() == 2);
  MatchingFlatSize(output_shape, incoming_shape, grad_shape);

  const int batches = incoming_shape.Dims(0);
  const int width = incoming_shape.Dims(1);

  // z_{i} -> softmax -> y_{i} -> (Loss function) -> Loss
  for (int b = 0; b < batches; ++b)
  {
    int b_offset = b * width;
    for (int w1 = 0; w1 < width; ++w1) // z_{i}
    {
      float sum = 0.0f;
      for (int w2 = 0; w2 < width; ++w2) // y_{i}
      {
        float val;
        if (w1 == w2)
        {
          val = output_data[b_offset + w2] * (1.f - output_data[b_offset + w2]);
        }
        else
        {
          val = -1 * output_data[b_offset + w2] * output_data[b_offset + w1];
        }
        val *= incoming_data[b_offset + w2];
        sum += val;
      }
      grad_data[b_offset + w1] = sum;
    }
  }
}

// inline void SoftMaxGrad(const Shape &output_shape, const float *output_data,
//   const Shape &incoming_shape, const float *incoming_data, const Shape &grad_shape, float
//   *grad_data)
// {
//   assert(output_shape.DimensionsCount() == 2);
//   MatchingFlatSize(output_shape, incoming_shape, grad_shape);

//   const auto output_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
//   const auto incoming_mat = MapAsMatrixWithLastDimAsRows(incoming_data, incoming_shape);
//   auto grad_mat = MapAsMatrixWithLastDimAsRows(grad_data, grad_shape);

//   const auto test_mat = output_mat.diagonal();
//   grad_mat = test_mat.array() - output_mat.array() * output_mat.transpose().array();
//   grad_mat *= incoming_mat;
//   // grad_mat -= output_mat.array() * output_mat.transpose().array();
// }

inline void SoftMaxGrad(const float *softmax_data, const float *input_data, const int input_size,
                        const int batch_size, float *output_data)
{
  assert(input_size > 0);

  for (int b = 0; b < batch_size; ++b)
  {
    for (int i = 0; i < input_size; ++i)
    {
      output_data[i] = softmax_data[i] * (1 - softmax_data[i]) * input_data[i];
    }

    // Advance in and out pointers for the next batch
    input_data += input_size;
    output_data += input_size;
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_SOFTMAX_H__
