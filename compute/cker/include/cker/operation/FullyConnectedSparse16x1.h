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
/* Copyright (c) 2018 Mozilla
                 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __NNFW_CKER_FULLY_CONNECTED_SPARSE16x1_H__
#define __NNFW_CKER_FULLY_CONNECTED_SPARSE16x1_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include "cker/TensorUtils.h"

namespace nnfw
{
namespace cker
{
inline void FullyConnectedSparseWeight16x1(const FullyConnectedParams &params,
                                           const Shape &input_shape, const float *input_data,
                                           const Shape &weights_shape, const float *weights_data,
                                           const Shape &bias_shape, const float *bias_data,
                                           const Shape &output_shape, float *output_data,
                                           const uint16_t *w1_segments, const uint16_t *w1_indices)
{
  UNUSED_RELEASE(input_shape);

  assert(weights_shape.DimensionsCount() == 2);
  assert(output_shape.DimensionsCount() == 2);

  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1);
  const int output_depth =
    MatchingDim(weights_shape, weights_dims_count - 2, output_shape, output_dims_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dims_count - 1);

  UNUSED_RELEASE(bias_shape);
  if (bias_data)
  {
    VectorBatchVectorAssign(bias_data, output_depth, batches, output_data);
  }
  else
  {
    ZeroVector(output_data, batches * output_depth);
  }
  for (int b = 0; b < batches; ++b)
  {
    int depth_size = output_depth / 16;
    for (int idx_0 = 0; idx_0 < depth_size; ++idx_0)
#ifdef USE_NEON
    {
      float *__restrict y;
      y = &output_data[b * output_depth + idx_0 * 16];
      /* keep y[0..15] in registers for duration of inner loop */
      float32x4_t y0_3 = vld1q_f32(&y[0]);
      float32x4_t y4_7 = vld1q_f32(&y[4]);
      float32x4_t y8_11 = vld1q_f32(&y[8]);
      float32x4_t y12_15 = vld1q_f32(&y[12]);
      for (auto pw1 = w1_segments[idx_0]; pw1 < w1_segments[idx_0 + 1]; ++pw1)
      {
        auto idx_1 = w1_indices[pw1];
        float32x4_t xj = vld1q_dup_f32(&input_data[b * accum_depth + idx_1]);
        float32x4_t wvec;

        wvec = vld1q_f32(&weights_data[0]);
        y0_3 = vmlaq_f32(y0_3, wvec, xj);
        wvec = vld1q_f32(&weights_data[4]);
        y4_7 = vmlaq_f32(y4_7, wvec, xj);
        wvec = vld1q_f32(&weights_data[8]);
        y8_11 = vmlaq_f32(y8_11, wvec, xj);
        wvec = vld1q_f32(&weights_data[12]);
        y12_15 = vmlaq_f32(y12_15, wvec, xj);

        weights_data += 16;
      }
      /* save y[0..15] back to memory */
      vst1q_f32(&y[0], y0_3);
      vst1q_f32(&y[4], y4_7);
      vst1q_f32(&y[8], y8_11);
      vst1q_f32(&y[12], y12_15);
    }
#else
    {
      for (auto pw1 = w1_segments[idx_0]; pw1 < w1_segments[idx_0 + 1]; ++pw1)
      {
        float *__restrict y;
        float xj;
        auto idx_1 = w1_indices[pw1];
        xj = input_data[b * accum_depth + idx_1];
        y = &output_data[b * output_depth + idx_0 * 16];
        y[0] += weights_data[0] * xj;
        y[1] += weights_data[1] * xj;
        y[2] += weights_data[2] * xj;
        y[3] += weights_data[3] * xj;
        y[4] += weights_data[4] * xj;
        y[5] += weights_data[5] * xj;
        y[6] += weights_data[6] * xj;
        y[7] += weights_data[7] * xj;
        y[8] += weights_data[8] * xj;
        y[9] += weights_data[9] * xj;
        y[10] += weights_data[10] * xj;
        y[11] += weights_data[11] * xj;
        y[12] += weights_data[12] * xj;
        y[13] += weights_data[13] * xj;
        y[14] += weights_data[14] * xj;
        y[15] += weights_data[15] * xj;
        weights_data += 16;
      }
    }
#endif
  }
  if (params.activation != FusedActivationFunctionType::kNone)
  {
    // Apply activation function
    ApplyActivationToVector(output_data, batches * output_depth, params.activation, output_data);
  }
}
} // namespace cker
} // namespace nnfw
#endif // __NNFW_CKER_FULLY_CONNECTED_SPARSE16x1_H__
