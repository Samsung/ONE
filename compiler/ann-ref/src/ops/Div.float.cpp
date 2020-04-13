/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#include "Div.float.h"

#include "internal/Array.h"
#include "internal/NDArray.h"
#include "internal/Matrix.h"
#include "internal/Fused.h"
#include "internal/ActivationUtils.h"

template <FusedActivationFunctionType Ac>
void Div(const float *input1_data, const Dims<4> &input1_dims, const float *input2_data,
         const Dims<4> &input2_dims, float *output_data, const Dims<4> &output_dims)
{
  MatchingArraySize(input1_dims, 3, input2_dims, 3, output_dims, 3);
  MatchingArraySize(input1_dims, 2, input2_dims, 2, output_dims, 2);
  MatchingArraySize(input1_dims, 1, input2_dims, 1, output_dims, 1);
  MatchingArraySize(input1_dims, 0, input2_dims, 0, output_dims, 0);
  DCHECK(IsPackedWithoutStrides(input1_dims));
  DCHECK(IsPackedWithoutStrides(input2_dims));
  DCHECK(IsPackedWithoutStrides(output_dims));

  const int size = input1_dims.sizes[3] * input1_dims.strides[3];

  for (int i = 0; i < size; i++)
  {
    auto x = input1_data[i] / input2_data[i];
    output_data[i] = ActivationFunction<Ac>(x);
  }
}

// From optimized_ops.h in TensorFlow Lite
//
// TODO: We can implement BroadcastDiv on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO: BroadcastDiv is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
template <FusedActivationFunctionType Ac>
void BroadcastDiv(const float *input1_data, const Dims<4> &input1_dims, const float *input2_data,
                  const Dims<4> &input2_dims, float *output_data, const Dims<4> &output_dims)
{
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < ArraySize(output_dims, 3); ++b)
  {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y)
    {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x)
      {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c)
        {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunction<Ac>(input1_data[SubscriptToIndex(desc1, c, x, y, b)] /
                                     input2_data[SubscriptToIndex(desc2, c, x, y, b)]);
        }
      }
    }
  }
}

bool divFloat32(const float *in1, const Shape &shape1, const float *in2, const Shape &shape2,
                int32_t activation, float *out, const Shape &shapeOut)
{
  bool needBroadcast = !SameShape(shape1, shape2);

#define ANDROID_NN_NORMAL_DIV(activation)                                        \
  Div<FusedActivationFunctionType::activation>(in1, convertShapeToDims(shape1),  \
                                               in2, convertShapeToDims(shape2),  \
                                               out, convertShapeToDims(shapeOut))

#define ANDROID_NN_BROADCAST_DIV(activation)              \
  BroadcastDiv<FusedActivationFunctionType::activation>(  \
      in1, convertShapeToDims(shape1),                    \
      in2, convertShapeToDims(shape2),                    \
      out, convertShapeToDims(shapeOut))

  if (needBroadcast)
  {
    ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_BROADCAST_DIV)
  }
  else
  {
    ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_NORMAL_DIV)
  }

#undef ANDROID_NN_NORMAL_ADD
#undef ANDROID_NN_BROADCAST_ADD
  return true;
}
