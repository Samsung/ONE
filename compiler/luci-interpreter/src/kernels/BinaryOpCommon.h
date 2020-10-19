/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LUCI_INTERPRETER_KERNELS_BINARYOPUTILS_H
#define LUCI_INTERPRETER_KERNELS_BINARYOPUTILS_H

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace luci_interpreter
{
namespace kernels
{

// Derived from tensorflow/lite/kernels/internal/reference/maximum_minimum.h (v2.3.0).
template <typename T, typename Op, int N = 5>
void BinaryOpBroadcastSlow(const tflite::RuntimeShape &unextended_input1_shape,
                           const T *input1_data,
                           const tflite::RuntimeShape &unextended_input2_shape,
                           const T *input2_data,
                           const tflite::RuntimeShape &unextended_output_shape, T *output_data,
                           Op op)
{
  if (unextended_input1_shape == unextended_input2_shape)
  {
    const int flat_size = tflite::MatchingElementsSize(
        unextended_input1_shape, unextended_input2_shape, unextended_output_shape);
    for (int i = 0; i < flat_size; ++i)
    {
      output_data[i] = op(input1_data[i], input2_data[i]);
    }
  }
  else
  {
    assert(unextended_input1_shape.DimensionsCount() <= N);
    assert(unextended_input2_shape.DimensionsCount() <= N);
    assert(unextended_output_shape.DimensionsCount() <= N);

    tflite::NdArrayDesc<N> desc1{};
    tflite::NdArrayDesc<N> desc2{};
    tflite::NdArrayDesc<N> output_desc{};
    tflite::NdArrayDescsForElementwiseBroadcast(unextended_input1_shape, unextended_input2_shape,
                                                &desc1, &desc2);
    tflite::CopyDimsToDesc(tflite::RuntimeShape::ExtendedShape(N, unextended_output_shape),
                           &output_desc);

    auto fn = [&](int indexes[N]) {
      output_data[SubscriptToIndex(output_desc, indexes)] =
          op(input1_data[SubscriptToIndex(desc1, indexes)],
             input2_data[SubscriptToIndex(desc2, indexes)]);
    };
    tflite::NDOpsHelper<N>(output_desc, fn);
  }
}

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_BINARYOPUTILS_H
