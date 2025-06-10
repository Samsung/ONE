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

#ifndef __NNFW_CKER_FULLY_CONNECTED_DENSE16x1_H__
#define __NNFW_CKER_FULLY_CONNECTED_DENSE16x1_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include "cker/TensorUtils.h"

namespace nnfw
{
namespace cker
{
#if defined(__aarch64__) && defined(USE_NEON)
inline void FullyConnected16x1Float32(const FullyConnectedParams &params, const Shape &input_shape,
                                      const float *input_data, const Shape &weights_shape,
                                      const float *weights_data, const Shape &,
                                      const float *bias_data, const Shape &, float *output_data)
{
  int total_input_size = input_shape.FlatSize();
  int input_size = weights_shape.Dims(1);
  const int batch_size = total_input_size / input_size;
  const int num_units = weights_shape.Dims(0);

  float *out = output_data;
  const float *weights = weights_data;
  int rows = num_units;
  int cols = input_size;
  int col_stride = input_size;
  const float *x = input_data;

  // Output = bias if bias tensor exists.
  if (bias_data)
  {
    VectorBatchVectorAssign(bias_data, num_units, batch_size, output_data);
  }
  else
  {
    ZeroVector(output_data, batch_size * num_units);
  }

  //  rows : out, cols : in
  int i, j;
  for (i = 0; i < rows; i += 16)
  {
    const float *w = &weights[i * col_stride];

    /* keep y[0..15] in registers for duration of inner loop */
    float *__restrict y = &out[i];

    float32x4_t y0_3 = vld1q_f32(&y[0]);
    float32x4_t y4_7 = vld1q_f32(&y[4]);
    float32x4_t y8_11 = vld1q_f32(&y[8]);
    float32x4_t y12_15 = vld1q_f32(&y[12]);

    for (j = 0; j < cols; j++)
    {
      float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;
      float32x4_t xj;

      xj = vld1q_dup_f32(&x[j]);

      wvec0_3 = vld1q_f32(&w[0]);
      y0_3 = vmlaq_f32(y0_3, wvec0_3, xj);
      wvec4_7 = vld1q_f32(&w[4]);
      y4_7 = vmlaq_f32(y4_7, wvec4_7, xj);
      wvec8_11 = vld1q_f32(&w[8]);
      y8_11 = vmlaq_f32(y8_11, wvec8_11, xj);
      wvec12_15 = vld1q_f32(&w[12]);
      y12_15 = vmlaq_f32(y12_15, wvec12_15, xj);

      w += 16;
    }

    /* save y[0..15] back to memory */

    vst1q_f32(&y[0], y0_3);
    vst1q_f32(&y[4], y4_7);
    vst1q_f32(&y[8], y8_11);
    vst1q_f32(&y[12], y12_15);
  }
  if (params.activation != FusedActivationFunctionType::kNone)
  {
    // Apply activation function
    ApplyActivationToVector(output_data, batch_size * num_units, params.activation, output_data);
  }
}
#endif
} // namespace cker
} // namespace nnfw
#endif // __NNFW_CKER_FULLY_CONNECTED_DENSE16x1_H__
