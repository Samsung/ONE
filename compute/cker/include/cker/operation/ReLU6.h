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

#ifndef __NNFW_CKER_RELU6_H__
#define __NNFW_CKER_RELU6_H__

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"

#include <cmath>
#include <Eigen/Core>

namespace nnfw
{
namespace cker
{

inline void ReLU6(const Shape &input_shape, const float *input_data, float *output_data)
{
  int size = input_shape.FlatSize();

  for (int i = 0; i < size; ++i)
  {
    if (input_data[i] <= 0)
    {
      output_data[i] = 0;
    }
    else if (input_data[i] > 6.0)
    {
      output_data[i] = 6.0;
    }
    else
    {
      output_data[i] = input_data[i];
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_RELU6_H__
