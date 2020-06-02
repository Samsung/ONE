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

#ifndef __NNFW_CKER_RANGE_H__
#define __NNFW_CKER_RANGE_H__

#include "cker/Shape.h"

#include <cmath>

namespace nnfw
{
namespace cker
{
template <typename T> inline int GetSize(T start, T limit, T delta)
{
  int size = (std::is_integral<T>::value
                  ? ((std::abs(*limit - *start) + std::abs(*delta) - 1) / std::abs(*delta))
                  : std::ceil(std::abs((*limit - *start) / *delta)));
  return size;
}
template <typename T>
inline void Range(const T start_data, const T limit_data, const T delta_data, T output_data)
{
  const int num_elements = GetSize<T>(start_data, limit_data, delta_data);
  T start = start_data;

  for (int i = 0; i < num_elements; ++i)
  {
    output_data[i] = *start;
    *start += *delta_data;
  }
}
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_RANGE_H__
