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
 * Copyright (c) 2016-2018 ARM Limited.
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
#define SUB(x, y) (x) - (y)

#if defined(OFF_IN) && defined(OFF_ALPHA) && defined(OFF_OUT) && defined(SCALE_IN) && \
    defined(SCALE_ALPHA) && defined(SCALE_OUT) && defined(VEC_SIZE)

#define VEC_FLOAT VEC_DATA_TYPE(float, VEC_SIZE)
#define VEC_INT VEC_DATA_TYPE(int, VEC_SIZE)
#define VEC_UCHAR VEC_DATA_TYPE(uchar, VEC_SIZE)
#define CONVERT_RTE(x, type) (convert_##type##_rte((x)))
#define CONVERT_DOWN(x, type) CONVERT_RTE(x, type)
#define SELECT_TYPE VEC_INT

/** Returns result of prelu function implemented as below:
 *  f(input) = alpha * input for input < 0, f(input) = input for input >= 0.
 *
 * @attention Data type can be passed using the -DDATA_TYPE_IN compile flag, e.g.
 *            -DDATA_TYPE_IN=uchar
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g.
 *            -DVEC_SIZE=16
 * @note Can only take uchar data types.
 *
 * @param[in]  input1_ptr                            Pointer to the source image. Supported Data
 *                                                   types : QASYMM8
 * @param[in]  input1_stride_x                       Stride of the source image in X dimension (in
 *                                                   bytes)
 * @param[in]  input1_step_x                         input1_stride_x * number of elements along X
 *                                                   processed per workitem(in bytes)
 * @param[in]  input1_stride_y                       Stride of the source image in Y dimension (in
 *                                                   bytes)
 * @param[in]  input1_step_y                         input1_stride_y * number of elements along Y
 *                                                   processed per workitem(in bytes)
 * @param[in]  input1_stride_z                       Stride of the source tensor in Z dimension (in
 *                                                   bytes)
 * @param[in]  input1_step_z                         input1_stride_z * number of elements along Z
 *                                                   processed per workitem(in bytes)
 * @param[in]  input1_offset_first_element_in_bytes  The offset of the first element in the source
 *                                                   image
 * @param[in]  alpha_ptr                             Pointer to the source image. Supported Data
 *                                                   types : QASYMM8
 * @param[in]  alpha_stride_x                        Stride of the source image in X dimension (in
 *                                                   bytes)
 * @param[in]  alpha_step_x                          input2_stride_x * number of elements along X
 *                                                   processed per workitem(in bytes)
 * @param[in]  alpha_stride_y                        Stride of the source image in Y dimension (in
 *                                                   bytes)
 * @param[in]  alpha_step_y                          input2_stride_y * number of elements along Y
 *                                                   processed per workitem(in bytes)
 * @param[in]  alpha_stride_z                        Stride of the source tensor in Z dimension (in
 *                                                   bytes)
 * @param[in]  alpha_step_z                          input2_stride_z * number of elements along Z
 *                                                   processed per workitem(in bytes)
 * @param[in]  alpha_offset_first_element_in_bytes   The offset of the first element in the source
 *                                                   image
 * @param[out] output_ptr                            Pointer to the destination image. Supported
 *                                                   data types: same as @p input_ptr
 * @param[in]  output_stride_x                       Stride of the destination image in X dimension
 *                                                   (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X
 *                                                   processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination image in Y dimension
 *                                                   (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y
 *                                                   processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the source tensor in Z dimension (in
 *                                                   bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z
 *                                                   processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the
 *                                                   destination image
 */
__kernel void prelu_qasymm8(TENSOR3D_DECLARATION(input), TENSOR3D_DECLARATION(alpha),
                            TENSOR3D_DECLARATION(output))
{
  // Get pixels pointer
  Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
  Tensor3D alpha = CONVERT_TO_TENSOR3D_STRUCT(alpha);
  Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

  VEC_INT in_vec = CONVERT(VLOAD(VEC_SIZE)(0, (__global uchar *)input.ptr), VEC_INT);
  VEC_INT alpha_vec = CONVERT(VLOAD(VEC_SIZE)(0, (__global uchar *)alpha.ptr), VEC_INT);

  in_vec = SUB(in_vec, (VEC_INT)((int)OFF_IN));
  alpha_vec = SUB(alpha_vec, (VEC_INT)((int)OFF_ALPHA));

  const VEC_FLOAT inf32 = CONVERT(in_vec, VEC_FLOAT) * (VEC_FLOAT)((float)SCALE_IN);
  const VEC_FLOAT alphaf32 = CONVERT(alpha_vec, VEC_FLOAT) * (VEC_FLOAT)((float)SCALE_ALPHA);
  const VEC_FLOAT outf32 =
      select(inf32, inf32 * alphaf32, CONVERT(inf32 < (VEC_FLOAT)0, SELECT_TYPE));
  const VEC_FLOAT qresf32 = outf32 / ((VEC_FLOAT)(float)SCALE_OUT) + ((VEC_FLOAT)((float)OFF_OUT));
  const VEC_UCHAR res = CONVERT_SAT(CONVERT_DOWN(qresf32, VEC_INT), VEC_UCHAR);

  VSTORE(VEC_SIZE)
  (res, 0, (__global uchar *)output.ptr);
}

#endif // defined(OFF_IN) && defined(OFF_ALPHA) && defined(OFF_OUT) && defined(SCALE_IN) &&
       // defined(SCALE_ALPHA) && defined(SCALE_OUT) && defined(VEC_SIZE)
