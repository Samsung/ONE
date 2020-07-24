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

#ifndef SCALE
#define SCALE 1.0f
#endif
#ifndef OFFSET
#define OFFSET 0
#endif
#ifndef VEC_SIZE
#define VEC_SIZE 1
#endif

#if defined(DATA_TYPE_IN) && defined(DATA_TYPE_OUT)
/** Perform a cast operation on an input tensor.
 *
 * @attention Data types of both input and output can be passed using the -DDATA_TYPE_IN and
 *            -DDATA_TYPE_OUT compile flag, e.g. -DDATA_TYPE_IN=float, -DDATA_TYPE_OUT=int
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g.
 *            -DVEC_SIZE=16
 * @attention -DBOOL_INPUT : Whether type of input is bool.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data
 *                                                  types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in
 *                                                  bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X
 *                                                  processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in
 *                                                  bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y
 *                                                  processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in
 *                                                  bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z
 *                                                  processed per workitem(in bytes)
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
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the
 *                                                  destination image
 */
__kernel void cast(TENSOR3D_DECLARATION(input), TENSOR3D_DECLARATION(output))
{
  Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
  Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

  VSTORE(VEC_SIZE)
  (CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_IN *)input.ptr),
           VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)),
   0, (__global DATA_TYPE_OUT *)output.ptr);
  VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)
  res = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_IN *)input.ptr),
                VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE));
#if defined(BOOL_INPUT)
  VEC_DATA_TYPE(char, VEC_SIZE) tmp = CONVERT(res, VEC_DATA_TYPE(char, VEC_SIZE));
  VEC_DATA_TYPE(char, VEC_SIZE) mask = (VEC_DATA_TYPE(char, VEC_SIZE))(1);
  res = CONVERT(tmp & mask, VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE));
#endif // defined(BOOL_INPUT)

  VSTORE(VEC_SIZE)(res, 0, (__global DATA_TYPE_OUT *)output.ptr);
}

/** Perform a cast operation on an QASYMM8 input tensor.
 * @attention Data types of both input and output can be passed using the -DDATA_TYPE_IN and
 *            -DDATA_TYPE_OUT compile flag, e.g. -DDATA_TYPE_IN=float, -DDATA_TYPE_OUT=int
 * @attention Offset and Scale of input should be given as a preprocessor argument using
 *            -DOFFSET=int, -DSCALE=float. e.g. -DOFFSET=1, -DSCALE=0.5
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g.
 *            -DVEC_SIZE=16
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data
 *                                                  types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in
 *                                                  bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X
 *                                                  processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in
 *                                                  bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y
 *                                                  processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in
 *                                                  bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z
 *                                                  processed per workitem(in bytes)
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
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the
 *                                                  destination image
 */
__kernel void cast_qasymm_in(TENSOR3D_DECLARATION(input), TENSOR3D_DECLARATION(output))
{
  Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
  Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

  VEC_DATA_TYPE(DATA_TYPE_IN, VEC_SIZE)
  in_data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_IN *)input.ptr);
  VEC_DATA_TYPE(int, VEC_SIZE) offset = (VEC_DATA_TYPE(int, VEC_SIZE))(OFFSET);
  VEC_DATA_TYPE(float, VEC_SIZE) scale = (VEC_DATA_TYPE(float, VEC_SIZE))(SCALE);

  VEC_DATA_TYPE(int, VEC_SIZE) tmp = CONVERT(in_data, VEC_DATA_TYPE(int, VEC_SIZE)) - offset;
  VEC_DATA_TYPE(float, VEC_SIZE) out_data = CONVERT(tmp, VEC_DATA_TYPE(float, VEC_SIZE)) * scale;

  VSTORE(VEC_SIZE)
  (CONVERT(out_data, VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)), 0,
   (__global DATA_TYPE_OUT *)output.ptr);
}

/** Perform a cast operation on an QASYMM8 output tensor.
 * @attention Data types of both input and output can be passed using the -DDATA_TYPE_IN and
 *            -DDATA_TYPE_OUT compile flag, e.g. -DDATA_TYPE_IN=float, -DDATA_TYPE_OUT=int
 * @attention Offset and Scale of output should be given as a preprocessor argument using
 *            -DOFFSET=int, -DSCALE=float. e.g. -DOFFSET=1, -DSCALE=0.5
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g.
 *            -DVEC_SIZE=16
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data
 *                                                  types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in
 *                                                 bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X
 *                                                  processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in
 *                                                  bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y
 *                                                  processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in
 *                                                  bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z
 *                                                  processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source
 *                                                  image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data
 *                                                  types: U8
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
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the
 *                                                  destination image
 */
__kernel void cast_qasymm_out(TENSOR3D_DECLARATION(input), TENSOR3D_DECLARATION(output))
{
  Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
  Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

  VEC_DATA_TYPE(DATA_TYPE_IN, VEC_SIZE)
  in_data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE_IN *)input.ptr);
  VEC_DATA_TYPE(int, VEC_SIZE) offset = (VEC_DATA_TYPE(int, VEC_SIZE))(OFFSET);
  VEC_DATA_TYPE(float, VEC_SIZE) scale = (VEC_DATA_TYPE(float, VEC_SIZE))(SCALE);

  VEC_DATA_TYPE(float, VEC_SIZE) tmp = CONVERT(in_data, VEC_DATA_TYPE(float, VEC_SIZE)) / scale;
  VEC_DATA_TYPE(float, VEC_SIZE) out_data = tmp + CONVERT(offset, VEC_DATA_TYPE(float, VEC_SIZE));

  VSTORE(VEC_SIZE)
  (CONVERT(out_data, VEC_DATA_TYPE(DATA_TYPE_OUT, VEC_SIZE)), 0,
   (__global DATA_TYPE_OUT *)output.ptr);
}
#endif // defined(DATA_TYPE_IN) && defined(DATA_TYPE_OUT)
