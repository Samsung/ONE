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

#ifndef __NNFW_CKER_FILL_H__
#define __NNFW_CKER_FILL_H__

#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{
template <typename T> inline void Fill(const T value_data, const Shape &output_shape, T output_data)
{
  int output_size = output_shape.FlatSize();
  for (int i = 0; i < output_size; i++)
  {
    output_data[i] = *value_data;
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_FILL_H__
