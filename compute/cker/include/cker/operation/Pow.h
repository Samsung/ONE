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

#ifndef __NNFW_CKER_POW_H__
#define __NNFW_CKER_POW_H__

#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{

template <typename T>
inline void powImpl(const Shape &input1_shape, const T *input1_data, const Shape &input2_shape,
                    const T *input2_data, const Shape &output_shape, T *output_data)
{
  const int flat_size = MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i)
  {
    output_data[i] = std::pow(input1_data[i], input2_data[i]);
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_POW_H__
