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

#ifndef __NNFW_CKER_ELEMENTWISE_H__
#define __NNFW_CKER_ELEMENTWISE_H__

#include "cker/eigen/Utils.h"
#include "cker/Shape.h"
#include "cker/Types.h"
#include <Eigen/Core>

namespace nnfw
{
namespace cker
{

inline void Sin(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                float *output_data)
{
  const int size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < size; i++)
  {
    output_data[i] = std::sin(input_data[i]);
  }
}

inline void Cos(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                float *output_data)
{
  const int size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < size; i++)
  {
    output_data[i] = std::cos(input_data[i]);
  }
}

inline void Abs(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                float *output_data)
{
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() = input_map.array().abs();
}

inline void Rsqrt(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                  float *output_data)
{
  const int size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < size; i++)
  {
    output_data[i] = 1.f / std::sqrt(input_data[i]);
  }
}

template <typename T>
inline void Neg(const Shape &input_shape, const T *input_data, const Shape &output_shape,
                T *output_data)
{
  const int size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < size; i++)
  {
    output_data[i] = -input_data[i];
  }
}

inline void Log(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                float *output_data)
{
  const int size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < size; i++)
  {
    output_data[i] = std::log(input_data[i]);
  }
}

inline void Floor(const Shape &input_shape, const float *input_data, const Shape &output_shape,
                  float *output_data)
{
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++)
  {
    output_data[i] = std::floor(input_data[i]);
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_ELEMENTWISE_H__
