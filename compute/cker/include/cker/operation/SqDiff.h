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

#ifndef __NNFW_CKER_REDUCESQDIFF_H__
#define __NNFW_CKER_REDUCESQDIFF_H__

#include "cker/Shape.h"
#include "cker/Utils.h"

namespace nnfw
{
namespace cker
{

#define SQDIFF(N)                                                                        \
  do                                                                                     \
  {                                                                                      \
    NdArrayDesc<N> input1_desc;                                                          \
    NdArrayDesc<N> input2_desc;                                                          \
    NdArrayDesc<N> output_desc;                                                          \
    SqDiffImpl<T, N>(input1_shape, input1_data, input2_shape, input2_data, output_shape, \
                     output_data, &input1_desc, &input2_desc, &output_desc);             \
  } while (0);

template <typename T, int N>
void SqDiffImpl(const Shape &input1_shape, const T *input1_data, const Shape &input2_shape,
                const T *input2_data, const Shape &output_shape, T *output_data,
                NdArrayDesc<N> *desc1_in, NdArrayDesc<N> *desc2_in, NdArrayDesc<N> *desc_out)
{
  std::vector<int> input_iter;
  input_iter.resize(N);
  const auto output_dims = output_shape.DimsData();

  // Copy dims to desc, calculating strides.
  CopyDimsToDesc<N>(output_shape, desc_out);
  NdArrayDescsForElementwiseBroadcast<N>(input1_shape, input2_shape, desc1_in, desc2_in);

  do
  {
    int input1_indx = SubscriptToIndexGeneric(desc1_in, input_iter.data());
    int input2_indx = SubscriptToIndexGeneric(desc2_in, input_iter.data());
    int output_indx = SubscriptToIndexGeneric(desc_out, input_iter.data());
    output_data[output_indx] = (input1_data[input1_indx] - input2_data[input2_indx]) *
                               (input1_data[input1_indx] - input2_data[input2_indx]);
  } while (NextIndex(N, output_dims, input_iter.data()));
}

template <typename T>
void SqDiff(const Shape &input1_shape, const T *input1_data, const Shape &input2_shape,
            const T *input2_data, const Shape &output_shape, T *output_data)
{
  assert(input1_shape.DimensionsCount() > 0 && input2_shape.DimensionsCount() > 0 &&
         output_shape.DimensionsCount() > 0);
  int outRank = output_shape.DimensionsCount();

  switch (outRank)
  {
    case 4:
      SQDIFF(4);
      break;

    case 3:
      SQDIFF(3);
      break;

    case 2:
      SQDIFF(2);
      break;

    case 1:
      SQDIFF(1);
      break;

    default:
      throw std::runtime_error("Support up to 4-D tensors at present");
      break;
  }
}
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_REDUCESQDIFF_H__
