/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_LEKAY_RELU_H__
#define __NNFW_CKER_LEKAY_RELU_H__

#include "cker/Shape.h"
#include "cker/Types.h"

#include <cmath>

namespace nnfw
{
namespace cker
{

inline void LeakyReLU(const LeakyReluParams &params, const Shape &input_shape,
                      const float *input_data, const Shape &output_shape, float *output_data)
{
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++)
  {
    const float val = input_data[i];
    // Note that alpha might be > 1 or < 0, so we don't use std::max here.
    output_data[i] = val > 0 ? val : val * params.alpha;
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_RELU_H__
