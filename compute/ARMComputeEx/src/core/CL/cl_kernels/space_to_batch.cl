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

#if defined(DATA_TYPE) && defined(DEPTH_OUT) && defined(BATCH_IN) && defined(HEIGHT_IN) && \
    defined(WIDTH_IN) && defined(ZERO_VALUE)
/** Perform space to batch with input of 4D and NCHW format
 *
 * @attention Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @attention Output tensor depth should be given as a preprocessor argument using -DDEPTH_OUT=size.
 *            e.g. -DDEPTH_OUT=16
 * @attention Input tensor batch should be given as a preprocessor argument using -DBATCH_IN=size.
 *            e.g. -DBATCH_IN=16
 * @attention Input tensor height should be given as a preprocessor argument using -DHEIGHT_IN=size.
 *            e.g. -DHEIGHT_IN=16
 * @attention Input tensor width should be given as a preprocessor argument using -DHEIGHT_IN=size.
 *            e.g. -DWIDTH_IN=16
 * @attention The value to be set by pad value using -DZERO_VALUE=value. e.g. -DZERO_VALUE=0
 *
 * @param[in]  input_ptr                                   Pointer to the source tensor. Supported
 *                                                         data types: U8/S8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                              Stride of the source tensor in X
 *                                                         dimension (in bytes)
 * @param[in]  input_step_x                                input_stride_x * number of elements along
 *                                                         X processed per workitem(in  bytes)
 * @param[in]  input_stride_y                              Stride of the source tensor in Y
 *                                                         dimension (in bytes)
 * @param[in]  input_step_y                                input_stride_y * number of elements along
 *                                                         Y processed per workitem(in  bytes)
 * @param[in]  input_stride_z                              Stride of the source tensor in Z
 *                                                         dimension (in bytes)
 * @param[in]  input_step_z                                input_stride_z * number of elements along
 *                                                         Z processed per workitem(in  bytes)
 * @param[in]  input_stride_w                              Stride of the destination tensor in W
 *                                                         dimension (in bytes)
 * @param[in]  input_step_w                                input_stride_w * number of elements along
 *                                                         W processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes         The offset of the first element in the
 *                                                         source tensor
 * @param[out] output_ptr                                  Pointer to the destination tensor.
 *                                                         Supported data types: same as @p
 *                                                         input_ptr
 * @param[in]  output_stride_x                             Stride of the destination tensor in X
 *                                                         dimension (in bytes)
 * @param[in]  output_step_x                               output_stride_x * number of elements
 *                                                         along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                             Stride of the destination tensor in Y
 * dimension (in bytes)
 * @param[in]  output_step_y                               output_stride_y * number of elements
 *                                                         along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                             Stride of the destination tensor in Z
 *                                                         dimension (in bytes)
 * @param[in]  output_step_z                               output_stride_z * number of elements
 *                                                         along Z processed per workitem(in bytes)
 * @param[in]  output_stride_w                             Stride of the destination tensor in W
 *                                                         dimension (in bytes)
 * @param[in]  output_step_w                               output_stride_w * number of elements
 *                                                         along W processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes        The offset of the first element in the
 *                                                         destination tensor
 * @param[in]  block_size_ptr                              Pointer to the source tensor. Supported
 *                                                         data types: S32
 * @param[in]  block_size_stride_x                         Stride of the source tensor in X
 *                                                         dimension (in bytes)
 * @param[in]  block_size_step_x                           block_size_stride_x * number of elements
 *                                                         along X processed per workitem(in  bytes)
 * @param[in]  block_size_offset_first_element_in_bytes    The offset of the first element in the
 *                                                         destination tensor
 * @param[in]  padding_size_ptr                            Pointer to the source tensor. Supported
 *                                                         data types: S32
 * @param[in]  padding_size_stride_x                       Stride of the source tensor in X
 *                                                         dimension (in bytes)
 * @param[in]  padding_size_step_x                         padding_size_stride_x * number of
 *                                                         elements along X processed per workitem
 *                                                         (in bytes)
 * @param[in]  padding_size_stride_y                       Stride of the source tensor in Y
 *                                                         dimension (in bytes)
 * @param[in]  padding_size_step_y                         padding_size_stride_y * number of
 *                                                         elements along Y processed per workitem
 *                                                         (in  bytes)
 * @param[in]  padding_size_offset_first_element_in_bytes  The offset of the first element in the
 *                                                         destination tensor
 */
__kernel void space_to_batch_4d_nchw(TENSOR4D_DECLARATION(input), TENSOR4D_DECLARATION(output),
                                     VECTOR_DECLARATION(block_size),
                                     IMAGE_DECLARATION(padding_size))
{
  Tensor4D in = CONVERT_TO_TENSOR4D_STRUCT(input, 0);
  Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT(output, DEPTH_OUT);

  int block_size_x = *((__global int *)(block_size_ptr));
  int block_size_y = *((__global int *)(block_size_ptr + block_size_stride_x));
  int shift_x = (get_global_id(2) / DEPTH_OUT / BATCH_IN) % block_size_x;
  int shift_y = (get_global_id(2) / DEPTH_OUT / BATCH_IN) / block_size_x;

  int in_index[4] = {
      0,
  };
  in_index[0] = get_global_id(0) * block_size_x + shift_x - *((__global int *)(padding_size_ptr));
  in_index[1] = get_global_id(1) * block_size_y + shift_y -
                *((__global int *)(padding_size_ptr + padding_size_stride_y));
  in_index[2] = get_global_id(2) % DEPTH_OUT;
  in_index[3] = (get_global_id(2) / DEPTH_OUT) % BATCH_IN;

  if (in_index[0] < 0 || in_index[0] >= WIDTH_IN || in_index[1] < 0 || in_index[1] >= HEIGHT_IN)
  {
    *((__global DATA_TYPE *)out.ptr) = (DATA_TYPE)ZERO_VALUE;
  }
  else
  {
    *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(
        &in, in_index[0], in_index[1], in_index[2], in_index[3]));
  }
}
#endif // defined(DATA_TYPE) && defined(DEPTH_OUT) && defined(BATCH_IN) && defined(HEIGHT_IN) &&
       // defined(WIDTH_IN) && defined(ZERO_VALUE)

#if defined(DATA_TYPE) && defined(HEIGHT_OUT) && defined(BATCH_IN) && defined(HEIGHT_IN) && \
    defined(WIDTH_IN) && defined(ZERO_VALUE) && defined(VEC_SIZE)
/** Perform space to batch with input of 4D and NHWC format
 *
 * @attention Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @attention Output tensor depth should be given as a preprocessor argument using
 *            -DHEIGHT_OUT=size. e.g. -DHEIGHT_OUT=16
 * @attention Input tensor batch should be given as a preprocessor argument using -DBATCH_IN=size.
 *            e.g. -DBATCH_IN=16
 * @attention Input tensor height should be given as a preprocessor argument using -DHEIGHT_IN=size.
 *            e.g. -DHEIGHT_IN=16
 * @attention Input tensor width should be given as a preprocessor argument using -DHEIGHT_IN=size.
 *            e.g. -DWIDTH_IN=16
 * @attention The value to be set by pad value using -DZERO_VALUE=value. e.g. -DZERO_VALUE=0
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g.
 *            -DVEC_SIZE=16
 *
 * @param[in]  input_ptr                                   Pointer to the source tensor. Supported
 *                                                         data types: U8/S8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                              Stride of the source tensor in X
 *                                                         dimension (in bytes)
 * @param[in]  input_step_x                                input_stride_x * number of elements along
 *                                                         X processed per workitem(in  bytes)
 * @param[in]  input_stride_y                              Stride of the source tensor in Y
 *                                                         dimension (in bytes)
 * @param[in]  input_step_y                                input_stride_y * number of elements along
 *                                                         Y processed per workitem(in  bytes)
 * @param[in]  input_stride_z                              Stride of the source tensor in Z
 *                                                         dimension (in bytes)
 * @param[in]  input_step_z                                input_stride_z * number of elements along
 *                                                         Z processed per workitem(in  bytes)
 * @param[in]  input_stride_w                              Stride of the destination tensor in W
 *                                                         dimension (in bytes)
 * @param[in]  input_step_w                                input_stride_w * number of elements along
 *                                                         W processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes         The offset of the first element in the
 *                                                         source tensor
 * @param[out] output_ptr                                  Pointer to the destination tensor.
 *                                                         Supported data types: same as @p
 *                                                         input_ptr
 * @param[in]  output_stride_x                             Stride of the destination tensor in X
 *                                                         dimension (in bytes)
 * @param[in]  output_step_x                               output_stride_x * number of elements
 *                                                         along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                             Stride of the destination tensor in Y
 *                                                         dimension (in bytes)
 * @param[in]  output_step_y                               output_stride_y * number of elements
 *                                                         along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                             Stride of the destination tensor in Z
 *                                                         dimension (in bytes)
 * @param[in]  output_step_z                               output_stride_z * number of elements
 *                                                         along Z processed per workitem(in bytes)
 * @param[in]  output_stride_w                             Stride of the destination tensor in W
 *                                                         dimension (in bytes)
 * @param[in]  output_step_w                               output_stride_w * number of elements
 *                                                         along W processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes        The offset of the first element in the
 *                                                         destination tensor
 * @param[in]  block_size_ptr                              Pointer to the source tensor. Supported
 *                                                         data types: S32
 * @param[in]  block_size_stride_x                         Stride of the source tensor in X
 *                                                         dimension (in bytes)
 * @param[in]  block_size_step_x                           block_size_stride_x * number of elements
 *                                                         along X processed per workitem(in  bytes)
 * @param[in]  block_size_offset_first_element_in_bytes    The offset of the first element in the
 *                                                         destination tensor
 * @param[in]  padding_size_ptr                            Pointer to the source tensor. Supported
 *                                                         data types: S32
 * @param[in]  padding_size_stride_x                       Stride of the source tensor in X
 *                                                         dimension (in bytes)
 * @param[in]  padding_size_step_x                         padding_size_stride_x * number of
 *                                                         elements along X processed per workitem
 *                                                         (in  bytes)
 * @param[in]  padding_size_stride_y                       Stride of the source tensor in Y
 *                                                         dimension (in bytes)
 * @param[in]  padding_size_step_y                         padding_size_stride_y * number of
 *                                                         elements along Y processed per workitem
 *                                                         (in bytes)
 * @param[in]  padding_size_offset_first_element_in_bytes  The offset of the first element in the
 *                                                         destination tensor
 */
__kernel void space_to_batch_4d_nhwc(TENSOR4D_DECLARATION(input), TENSOR4D_DECLARATION(output),
                                     VECTOR_DECLARATION(block_size),
                                     IMAGE_DECLARATION(padding_size))
{
  Tensor4D in = CONVERT_TO_TENSOR4D_STRUCT(input, 0);
  Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT(output, HEIGHT_OUT);

  int block_size_x = *((__global int *)(block_size_ptr));
  int block_size_y = *((__global int *)(block_size_ptr + block_size_stride_x));
  int shift_x = (get_global_id(2) / HEIGHT_OUT / BATCH_IN) % block_size_x;
  int shift_y = (get_global_id(2) / HEIGHT_OUT / BATCH_IN) / block_size_x;

  int in_index[4] = {
      0,
  };
  in_index[0] = get_global_id(0) * VEC_SIZE;
  in_index[1] = get_global_id(1) * block_size_x + shift_x - *((__global int *)(padding_size_ptr));
  in_index[2] = get_global_id(2) % HEIGHT_OUT * block_size_y + shift_y -
                *((__global int *)(padding_size_ptr + padding_size_stride_y));
  in_index[3] = (get_global_id(2) / HEIGHT_OUT) % BATCH_IN;

  if (in_index[1] < 0 || in_index[1] >= WIDTH_IN || in_index[2] < 0 || in_index[2] >= HEIGHT_IN)
  {
    VSTORE(VEC_SIZE)
    ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))ZERO_VALUE, 0, (__global DATA_TYPE *)out.ptr);
  }
  else
  {
    VSTORE(VEC_SIZE)
    (CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)tensor4D_offset(&in, in_index[0], in_index[1],
                                                                      in_index[2], in_index[3])),
             VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)),
     0, (__global DATA_TYPE *)out.ptr);
  }
}

#endif // defined(DATA_TYPE) && defined(HEIGHT_OUT) && defined(BATCH_IN) && defined(HEIGHT_IN) &&
       // defined(WIDTH_IN) && defined(ZERO_VALUE) && defined(VEC_SIZE)
