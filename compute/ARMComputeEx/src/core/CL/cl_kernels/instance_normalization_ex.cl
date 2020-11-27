/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

/*
 * Copyright (c) 2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "helpers.h"

#if defined(VEC_SIZE) && defined(DATA_TYPE) && defined(EPSILON) && defined(DIM_X) && \
  defined(DIM_Y) && defined(DIM_Z)
/** This function normalizes the input 2D tensor across the first dimension with respect to mean and
 * standard deviation of the same dimension.
 *
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g.
 * -DVEC_SIZE=16
 * @attention Data type should be passed using the -DDATA_TYPE=data_type compile flag, e.g.
 * -DDATA_TYPE=float
 * @attention Normalization epsilon parameter should be given as a preprocessor argument with
 * -DEPSILON=value. e.g. -DEPSILON=0.001f
 * @attention Dimensions X, Y, and Z should be given as a preprocessor argument with -DDIM_X=value,
 * -DDIM_Y=value, -DDIM_Z=value. e.g. -DDIM_X=6, -DDIM_Y=2, -DDIM_Z=7
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported
 * data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the first source tensor in X dimension
 * (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X
 * processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the first source tensor in Y dimension
 * (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y
 * processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the first source tensor in Z dimension
 * (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z
 * processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first
 * source tensor
 * @param[out] output_ptr                           (Optional) Pointer to the destination tensor.
 * Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      (Optional) Stride of the destination tensor in X
 * dimension (in bytes)
 * @param[in]  output_step_x                        (Optional) output_stride_x * number of elements
 * along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      (Optional) Stride of the destination tensor in Y
 * dimension (in bytes)
 * @param[in]  output_step_y                        (Optional) output_stride_y * number of elements
 * along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      (Optional) Stride of the destination tensor in Z
 * dimension (in bytes)
 * @param[in]  output_step_z                        (Optional) output_stride_z * number of elements
 * along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes (Optional) The offset of the first element in
 * the destination tensor
 * @param[in]  gamma_ptr                            (Optional) Pointer to the gamma tensor.
 * Supported data types: same as @p input_ptr
 * @param[in]  gamma_stride_x                       (Optional) Stride of the gamma tensor in X
 * dimension (in bytes)
 * @param[in]  gamma_step_x                         (Optional) output_stride_x * number of elements
 * along X processed per workitem(in bytes)
 * @param[in]  gamma_offset_first_element_in_bytes  (Optional) The offset of the first element in
 * the gamma tensor
 * @param[in]  beta_ptr                             (Optional) Pointer to the beta tensor. Supported
 * data types: same as @p input_ptr
 * @param[in]  beta_stride_x                        (Optional) Stride of the beta tensor in X
 * dimension (in bytes)
 * @param[in]  beta_step_x                          (Optional) output_stride_x * number of elements
 * along X processed per workitem(in bytes)
 * @param[in]  beta_offset_first_element_in_bytes   (Optional) The offset of the first element in
 * the beta tensor
 */
__kernel void instance_normalization_ex(TENSOR4D_DECLARATION(input),
#ifndef IN_PLACE
                                        TENSOR4D_DECLARATION(output)
#endif /* IN_PLACE */
#ifdef GAMMA
                                          ,
                                        VECTOR_DECLARATION(gamma)
#endif // GAMMA
#ifdef BETA
                                          ,
                                        VECTOR_DECLARATION(beta)
#endif // BETA
)
{
  Tensor4D in = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, 0);
#ifndef IN_PLACE
  Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(output, 0);
#endif /* IN_PLACE */

  float sum = 0.f;
  float sum_sq = 0.f;

#if defined(NHWC)

  const int ch = get_global_id(0);    // Current channel
  const int batch = get_global_id(2); // Current batch
  const int elements_plane = DIM_Y * DIM_Z;

  for (int i_w = 0; i_w < DIM_Y; ++i_w)
  {
    for (int i_h = 0; i_h < DIM_Z; ++i_h)
    {
      float data = (float)*((__global DATA_TYPE *)tensor4D_offset(&in, ch, i_w, i_h, batch));
      sum += data;
      sum_sq += data * data;
    }
  }

#else // !defined(NHWC)
  const int ch = get_global_id(2) % DIM_Z;    // Current channel
  const int batch = get_global_id(2) / DIM_Z; // Current batch
  const int elements_plane = DIM_X * DIM_Y;

  VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
  part_sum = 0.f;
  VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
  part_sum_sq = 0.f;
  // Calculate partial sum
  for (int y = 0; y < DIM_Y; ++y)
  {
    int x = 0;
    for (; x <= (DIM_X - VEC_SIZE); x += VEC_SIZE)
    {
      // Load data
      VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
      data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)tensor4D_offset(&in, x, y, ch, batch));
      part_sum += data;
      part_sum_sq += data * data;
    }
    // Left-overs loop
    for (; x < DIM_X; ++x)
    {
      DATA_TYPE data = *((__global DATA_TYPE *)tensor4D_offset(&in, x, y, ch, batch));
      part_sum.s0 += data;
      part_sum_sq.s0 += data * data;
    }
  }
// Perform reduction
#if VEC_SIZE > 8
  part_sum.s01234567 += part_sum.s89abcdef;
  part_sum_sq.s01234567 += part_sum_sq.s89abcdef;
#endif // VEC_SIZE > 8
#if VEC_SIZE > 4
  part_sum.s0123 += part_sum.s4567;
  part_sum_sq.s0123 += part_sum_sq.s4567;
#endif // VEC_SIZE > 4
#if VEC_SIZE > 2
  part_sum.s01 += part_sum.s23;
  part_sum_sq.s01 += part_sum_sq.s23;
#endif // VEC_SIZE > 2
  part_sum.s0 += part_sum.s1;
  part_sum_sq.s0 += part_sum_sq.s1;

  sum = (float)part_sum.s0;
  sum_sq = (float)part_sum_sq.s0;

#endif // defined(NHWC)

  const float mean_float = (sum / elements_plane);
  const DATA_TYPE mean = (DATA_TYPE)mean_float;
  const float var_float = (sum_sq / elements_plane) - (mean_float * mean_float);
#if defined(GAMMA)
  const float multip_float = *((__global DATA_TYPE *)gamma_ptr + ch) / sqrt(var_float + EPSILON);
  const DATA_TYPE multip = (DATA_TYPE)multip_float;
#else  // !defined(GAMMA)
  const DATA_TYPE multip = (DATA_TYPE)0;
#endif // defined(GAMMA)
#if defined(BETA)
  const DATA_TYPE beta = *((__global DATA_TYPE *)beta_ptr + ch);
#else  // !defined(BETA)
  const DATA_TYPE beta = 0;
#endif // defined(BETA)

#if defined(NHWC)

  for (int i_w = 0; i_w < DIM_Y; ++i_w)
  {
    for (int i_h = 0; i_h < DIM_Z; ++i_h)
    {
      __global DATA_TYPE *input_address =
        (__global DATA_TYPE *)tensor4D_offset(&in, ch, i_w, i_h, batch);
#ifdef IN_PLACE
      __global DATA_TYPE *output_address = input_address;
#else  /* !IN_PLACE */
      __global DATA_TYPE *output_address =
        (__global DATA_TYPE *)tensor4D_offset(&out, ch, i_w, i_h, batch);
#endif /* IN_PLACE */
      *(output_address) = (*(input_address)-mean) * multip + beta;
    }
  }

#else // !defined(NHWC)
  for (int y = 0; y < DIM_Y; ++y)
  {
    int x = 0;
    for (; x <= (DIM_X - VEC_SIZE); x += VEC_SIZE)
    {
      __global DATA_TYPE *input_address =
        (__global DATA_TYPE *)tensor4D_offset(&in, x, y, ch, batch);
#ifdef IN_PLACE
      __global DATA_TYPE *output_address = input_address;
#else  /* !IN_PLACE */
      __global DATA_TYPE *output_address =
        (__global DATA_TYPE *)tensor4D_offset(&out, x, y, ch, batch);
#endif /* IN_PLACE */

      VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
      data = VLOAD(VEC_SIZE)(0, input_address);

      VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
      res = (data - mean) * multip + beta;
      VSTORE(VEC_SIZE)
      (res, 0, output_address);
    }
    // Left-overs loop
    for (; x < DIM_X; ++x)
    {
      __global DATA_TYPE *input_address =
        (__global DATA_TYPE *)tensor4D_offset(&in, x, y, ch, batch);
#ifdef IN_PLACE
      __global DATA_TYPE *output_address = input_address;
#else  /* !IN_PLACE */
      __global DATA_TYPE *output_address =
        (__global DATA_TYPE *)tensor4D_offset(&out, x, y, ch, batch);
#endif /* IN_PLACE */
      *(output_address) = (*(input_address)-mean) * multip + beta;
    }
  }
#endif // defined(NHWC)
}
#endif /* defined(VEC_SIZE) && defined(DATA_TYPE) && defined(EPSILON) && defined(DIM_X) && \
          defined(DIM_Y) && defined(DIM_Z) */
