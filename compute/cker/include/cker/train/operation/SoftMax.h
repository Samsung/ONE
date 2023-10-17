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

#include "cker/Shape.h"
// #include "cker/Utils.h"
// #include "cker/Types.h"
// #include "cker/eigen/Utils.h"

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

inline void SoftMaxGrad(const Shape &input_shape, const float *input_data,
                 const Shape &output_shape, float *output_data)
{
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i)
  {
    output_data[i] = input_data[i]*(1-input_data[i]);
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_SOFTMAX_H__
