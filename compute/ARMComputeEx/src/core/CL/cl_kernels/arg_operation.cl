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

#if defined(DATA_TYPE) && defined(DEPTH_OUT) && defined(OP_CODE)
/** Perform arg_max/arg_min
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type.
 *       e.g. -DDATA_TYPE=short
 * @attention Output tensor depth should be given as a preprocessor argument using -DDEPTH_OUT=size.
 *            e.g. -DDEPTH_OUT=16
 * @attention Operation type(code) specifying which operation to perform should be passed as
 *            preprocessor argument using -DOP_CODE = number. e.g. -DOP_CODE=1
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data
 * types:
 *                                                  U8/QASYMM8/S8/U16/S16/F16/U32/S32/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension
 *                                                  (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X
 *                                                  processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension
 *                                                  (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y
 *                                                  processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension
 *                                                  (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z
 *                                                  processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element
 *                                                  in the source image
 * @param[in]  input_stride_w                       Stride of the source tensor in W dimension
 *                                                  (in bytes)
 * @param[in]  input_step_w                         output_stride_w * number of elements along W
 *                                                  processed per workitem(in bytes)
 * @param[out] output_ptr                           Pointer to the destination image.
 *                                                  Supported data types: U32
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension
 *                                                  (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension
 *                                                  (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension
 *                                                  (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_stride_w                      Stride of the source tensor in W dimension
 *                                                  (in bytes)
 * @param[in]  output_step_w                        output_stride_w * number of elements along W
 *                                                  processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the
 *                                                  destination image
 * @param[in]  axis                                 Axis through which reduction occurs
 * @param[in]  dim                                  Dimension across the axis to be reduced.
 */

__kernel void arg_op(TENSOR4D_DECLARATION(input), TENSOR4D_DECLARATION(output), const int axis,
                     const int dim)
{
  Tensor4D in = CONVERT_TO_TENSOR4D_STRUCT(input, 0);
  Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT(output, DEPTH_OUT);

  int indices[4] = {
      get_global_id(0), get_global_id(1), get_global_id(2) % DEPTH_OUT,
      get_global_id(2) / DEPTH_OUT,
  };

  DATA_TYPE value =
      *((__global DATA_TYPE *)tensor4D_offset(&in, indices[0], indices[1], indices[2], indices[3]));
  DATA_TYPE tval = value;
  int idx = 0;
  for (int i = 1; i < dim; ++i)
  {
    indices[axis] = i;

#if OP_CODE == 1 // ArgMax
    value = max(value, *((__global DATA_TYPE *)tensor4D_offset(&in, indices[0], indices[1],
                                                               indices[2], indices[3])));
#elif OP_CODE == 2 // ArgMin
    value = min(value, *((__global DATA_TYPE *)tensor4D_offset(&in, indices[0], indices[1],
                                                               indices[2], indices[3])));
#else
    return;

#endif

    if (tval != value)
    {
      idx = indices[axis];
      tval = value;
    }
  }

  *((__global uint *)out.ptr) = idx;
}
#endif // defined(DATA_TYPE) && defined(DEPTH_OUT) && defined(OP_CODE)
