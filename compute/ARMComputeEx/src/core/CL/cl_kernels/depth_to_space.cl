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
 * Copyright (c) 2016, 2017 ARM Limited.
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

#if defined(DATA_TYPE) && defined(DEPTH_OUT) && defined(BLOCK_SIZE) && defined(Z_OUT)
/** Perform space to depth rearrangement of tensor
 *
 * @attention Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @attention Input tensor depth should be given as a preprocessor argument using -DDEPTH_OUT=size.
 *            e.g. -DDEPTH_OUT=16
 * @attention The value of the z-axis of output tensor should be given as a preprocessor argument
 *            using -DZ_OUT=size. e.g. -DZ_OUT=16
 * @attention block size should be given as a preprocessor argument using -DBLOCK_SIZE=size. e.g.
 *            -DBLOCK_SIZE=1
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data
 *                                                  types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in
 *                                                  bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X
 *                                                  processed per workitem(in  bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in
 *                                                  bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y
 *                                                  processed per workitem(in  bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in
 *                                                  bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z
 *                                                  processed per workitem(in  bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source
 *                                                  image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data
 *                                                  types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension
 *                                                  (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension
 *                                                  (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in
 *                                                  bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_stride_w                      Stride of the source tensor in W dimension (in
 *                                                  bytes)
 * @param[in]  output_step_w                        output_stride_w * number of elements along W
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the
 *                                                  destination image
 */
__kernel void depth_to_space_nchw(TENSOR4D_DECLARATION(input), TENSOR4D_DECLARATION(output))
{
  Tensor4D in = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, 0);
  Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT(output, Z_OUT);

  int out_index[4] = {0};
  int in_index[4] = {0};

  out_index[0] = get_global_id(0);         // W
  out_index[1] = get_global_id(1);         // H
  out_index[2] = get_global_id(2) % Z_OUT; // C
  out_index[3] = get_global_id(2) / Z_OUT; // B

  in_index[0] = out_index[0] / BLOCK_SIZE;
  in_index[1] = out_index[1] / BLOCK_SIZE;
  in_index[2] = out_index[2] +
                ((out_index[1] % BLOCK_SIZE) * BLOCK_SIZE + out_index[0] % BLOCK_SIZE) * DEPTH_OUT;
  in_index[3] = out_index[3];

  *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(
      &in, in_index[0], in_index[1], in_index[2], in_index[3]));
}
#endif // defined(DATA_TYPE) && defined(DEPTH_OUT) && defined(BLOCK_SIZE) && defined(Z_OUT)

#if defined(DATA_TYPE) && defined(DEPTH_OUT) && defined(BLOCK_SIZE) && defined(Z_OUT)
/** Perform space to depth rearrangement of tensor (NHWC)
 *
 * @attention Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @attention Output tensor depth should be given as a preprocessor argument using -DDEPTH_OUT=size.
 *            e.g. -DDEPTH_OUT=16
 * @attention The value of the z-axis of output tensor should be given as a preprocessor argument
 *            using -DZ_OUT=size. e.g. -DZ_OUT=16
 * @attention block size should be given as a preprocessor argument using -DBLOCK_SIZE=size. e.g.
 *            -DBLOCK_SIZE=1
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data
 *                                                  types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in
 *                                                  bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X
 *                                                  processed per workitem(in  bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in
 *                                                  bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y
 *                                                  processed per workitem(in  bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in
 *                                                  bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z
 *                                                  processed per workitem(in  bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source
 *                                                  image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data
 *                                                  types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension
 *                                                  (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension
 *                                                  (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in
 *                                                  bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_stride_w                      Stride of the source tensor in W dimension (in
 *                                                  bytes)
 * @param[in]  output_step_w                        output_stride_w * number of elements along W
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the
 *                                                  destination image
 */
__kernel void depth_to_space_nhwc(TENSOR4D_DECLARATION(input), TENSOR4D_DECLARATION(output))
{
  Tensor4D in = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, 0);
  Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT(output, Z_OUT);

  int out_index[4] = {0};
  int in_index[4] = {0};

  out_index[0] = get_global_id(0);         // C
  out_index[1] = get_global_id(1);         // W
  out_index[2] = get_global_id(2) % Z_OUT; // H
  out_index[3] = get_global_id(2) / Z_OUT; // B

  in_index[0] = out_index[0] +
                ((out_index[2] % BLOCK_SIZE) * BLOCK_SIZE + out_index[1] % BLOCK_SIZE) * DEPTH_OUT;
  in_index[1] = out_index[1] / BLOCK_SIZE;
  in_index[2] = out_index[2] / BLOCK_SIZE;
  in_index[3] = out_index[3];

  *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(
      &in, in_index[0], in_index[1], in_index[2], in_index[3]));
}
#endif // defined(DATA_TYPE) && defined(DEPTH_OUT) && defined(BLOCK_SIZE) && defined(Z_OUT)
