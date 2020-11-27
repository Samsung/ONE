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

#ifndef __NNFW_CKER_SLICE_H__
#define __NNFW_CKER_SLICE_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"

namespace nnfw
{
namespace cker
{

template <typename T>
inline void Slice(const SliceParams &op_params, const Shape &input_shape,
                  SequentialTensorWriter<T> *writer)
{
  // TODO(dkalenichenko): This op only supports 4D tensors or smaller.
  assert(op_params.begin_count <= 4);
  assert(op_params.size_count <= 4);

  const int begin_count = op_params.begin_count;
  const int size_count = op_params.size_count;
  // We front-pad the begin and size vectors.
  const int start_b = 4 - begin_count > 0 ? 0 : op_params.begin[0];
  const int stop_b = (4 - size_count > 0 || op_params.size[0] == -1) ? input_shape.Dims(0)
                                                                     : start_b + op_params.size[0];
  const int start_h = begin_count < 3 ? 0 : op_params.begin[begin_count - 3];
  const int stop_h = (size_count < 3 || op_params.size[size_count - 3] == -1)
                       ? input_shape.Dims(1)
                       : start_h + op_params.size[size_count - 3];
  const int start_w = begin_count < 2 ? 0 : op_params.begin[begin_count - 2];
  const int stop_w = (size_count < 2 || op_params.size[size_count - 2] == -1)
                       ? input_shape.Dims(2)
                       : start_w + op_params.size[size_count - 2];
  const int start_d = begin_count < 1 ? 0 : op_params.begin[begin_count - 1];
  const int stop_d = (size_count < 1 || op_params.size[size_count - 1] == -1)
                       ? input_shape.Dims(3)
                       : start_d + op_params.size[size_count - 1];

  for (int in_b = start_b; in_b < stop_b; ++in_b)
  {
    for (int in_h = start_h; in_h < stop_h; ++in_h)
    {
      for (int in_w = start_w; in_w < stop_w; ++in_w)
      {
        const int len = stop_d - start_d;
        if (len > 0)
          writer->WriteN(Offset(input_shape, in_b, in_h, in_w, start_d), len);
      }
    }
  }
}

template <typename T>
inline void Slice(const SliceParams &op_params, const Shape &input_shape, const T *input_data,
                  T *output_data)
{
  SequentialTensorWriter<T> writer(input_data, output_data);
  return Slice(op_params, input_shape, &writer);
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_SLICE_H__
