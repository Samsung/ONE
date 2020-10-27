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

#ifndef __NNFW_CKER_ADDN_H__
#define __NNFW_CKER_ADDN_H__

#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{

template <typename T>
void AddN(const Shape &input_shape, const size_t num_inputs, const T **input_data, T *output_data)
{
  const size_t size = input_shape.FlatSize();
  for (size_t i = 0; i < size; ++i)
  {
    T x = 0;
    for (size_t j = 0; j < num_inputs; ++j)
    {
      x += input_data[j][i];
    }
    output_data[i] = x;
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_ADDN_H__
