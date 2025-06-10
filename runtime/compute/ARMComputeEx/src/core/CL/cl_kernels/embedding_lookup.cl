/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * Copyright (c) 2017 ARM Limited.
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

#ifndef VEC_SIZE
#define VEC_SIZE 1
#endif

#if defined(DATA_TYPE) && defined(DEPTH_OUT) && defined(NUM_DIMS)
/** Perform embedding_lookup of input tensor
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g.
 *       -DDATA_TYPE=short
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g.
 *            -DVEC_SIZE=16
 * @attention Output tensor depth should be given as a preprocessor argument using
 *            -DDEPTH_OUT=depth. e.g. -DDEPTH_OUT=16
 * @attention Number of input dimensions are passed as a preprocessor argument using
 *            -DNUM_DIMS=size, e.g. -DNUM_DIMS=4
 *
 * @param[in]  input_ptr                             Pointer to the source tensor. Supported data
 *                                                   types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                        Stride of the source tensor in X dimension (in
 *                                                   bytes)
 * @param[in]  input_step_x                          input_stride_x * number of elements along X
 *                                                   processed per workitem(in bytes)
 * @param[in]  input_stride_y                        Stride of the source tensor in Y dimension (in
 *                                                   bytes)
 * @param[in]  input_step_y                          input_stride_y * number of elements along Y
 *                                                   processed per workitem(in bytes)
 * @param[in]  input_stride_z                        Stride of the source tensor in Z dimension (in
 *                                                   bytes)
 * @param[in]  input_step_z                          input_stride_z * number of elements along Z
 *                                                   processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes   The offset of the first element in the source
 *                                                   tensor
 * @param[in]  input_stride_w                        Stride of the source tensor in W dimension (in
 *                                                   bytes)
 * @param[in]  input_step_w                          output_stride_w * number of elements along W
 *                                                   processed per workitem(in bytes)
 * @param[out] output_ptr                            Pointer to the destination tensor. Supported
 *                                                   data types: same as @p input_ptr
 * @param[in]  output_stride_x                       Stride of the destination tensor in X dimension
 *                                                   (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X
 *                                                   processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination tensor in Y dimension
 *                                                   (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y
 *                                                   processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the source tensor in Z dimension (in
 *                                                   bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z
 *                                                   processed per workitem(in bytes)
 * @param[in]  output_stride_w                       Stride of the source tensor in W dimension (in
 *                                                   bytes)
 * @param[in]  output_step_w                         output_stride_w * number of elements along W
 *                                                   processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the
 *                                                   destination tensor
 * @param[in]  lookups_ptr                           Pointer to the lookups vector. Supported data
 *                                                   types: S32
 * @param[in]  lookups_stride_x                      Stride of the lookups vector in X dimension (in
 *                                                   bytes)
 * @param[in]  lookups_step_x                        lookups_stride_x * number of elements along X
 *                                                   processed per workitem(in bytes)
 * @param[in]  lookups_offset_first_element_in_bytes The offset of the first element in the lookups
 *                                                   vector
 */

__kernel void embedding_lookup(TENSOR4D_DECLARATION(input), TENSOR4D_DECLARATION(output),
                               VECTOR_DECLARATION(lookups))
{
  Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT(output, DEPTH_OUT);
  Tensor4D in = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, DEPTH_OUT);

  Vector lups = CONVERT_TO_VECTOR_STRUCT_NO_STEP(lookups);

  // lookup ids for based on the tensor dimensions
  int lup_id[4] = {0};

  lup_id[0] =
    (NUM_DIMS == 1) ? *((__global int *)vector_offset(&lups, get_global_id(0))) : get_global_id(0);
  lup_id[1] =
    (NUM_DIMS == 2) ? *((__global int *)vector_offset(&lups, get_global_id(1))) : get_global_id(1);
  lup_id[2] = (NUM_DIMS == 3) ? *((__global int *)vector_offset(&lups, get_global_id(2)))
                              : get_global_id(2) % DEPTH_OUT;
  lup_id[3] = (NUM_DIMS == 4)
                ? *((__global int *)vector_offset(&lups, get_global_id(2) / DEPTH_OUT))
                : get_global_id(2) / DEPTH_OUT;

  in.ptr += input_offset_first_element_in_bytes + lup_id[0] * input_step_x +
            lup_id[1] * input_step_y + lup_id[2] * input_step_z + lup_id[3] * input_step_w;

  VSTORE(VEC_SIZE)
  (CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in.ptr), VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)), 0,
   (__global DATA_TYPE *)out.ptr);
}
#endif // defined(DATA_TYPE) && defined(DEPTH_OUT) && defined(NUM_DIMS)
