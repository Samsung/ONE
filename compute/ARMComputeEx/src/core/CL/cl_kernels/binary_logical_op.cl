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

#ifndef VEC_SIZE
#define VEC_SIZE 1
#endif

#if defined(OP_CODE) && defined(DATA_TYPE)
/** returns truth value of the two input tensors for BINARY LOGICAL OP.
 *  where BINARY LOGICAL OP can be AND, OR.
 *
 * @attention Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=uchar
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size.
 *            e.g. -DVEC_SIZE=16
 * @attention Operation type(code) specifying which operation to perform should be passed as
 *            preprocessor argument using -DOP_CODE = number. e.g. -DOP_CODE=1
 *
 * @param[in]  input1_ptr                            Pointer to the source tensor.
 *                                                   Supported data types: QASYMM8
 * @param[in]  input1_stride_x                       Stride of the source tensor in X dimension
 *                                                   (in bytes)
 * @param[in]  input1_step_x                         input1_stride_x * number of elements along X
 *                                                   processed per workitem(in bytes)
 * @param[in]  input1_stride_y                       Stride of the source tensor in Y dimension
 *                                                   (in bytes)
 * @param[in]  input1_step_y                         input1_stride_y * number of elements along Y
 *                                                   processed per workitem(in bytes)
 * @param[in]  input1_stride_z                       Stride of the source tensor in Z dimension
 *                                                   (in bytes)
 * @param[in]  input1_step_z                         input1_stride_z * number of elements along Z
 *                                                   processed per workitem(in bytes)
 * @param[in]  input1_offset_first_element_in_bytes  The offset of the first element in the source
 *                                                   tensor
 * @param[in]  input2_ptr                            Pointer to the source tensor.
 *                                                   Supported data types: QASYMM8
 * @param[in]  input2_stride_x                       Stride of the source tensor in X dimension
 *                                                   (in bytes)
 * @param[in]  input2_step_x                         input2_stride_x * number of elements along X
 *                                                   processed per workitem(in bytes)
 * @param[in]  input2_stride_y                       Stride of the source tensor in Y dimension
 *                                                   (in bytes)
 * @param[in]  input2_step_y                         input2_stride_y * number of elements along Y
 *                                                   processed per workitem(in bytes)
 * @param[in]  input2_stride_z                       Stride of the source tensor in Z dimension
 *                                                   (in bytes)
 * @param[in]  input2_step_z                         input2_stride_z * number of elements along Z
 *                                                   processed per workitem(in bytes)
 * @param[in]  input2_offset_first_element_in_bytes  The offset of the first element in the source
 *                                                   tensor
 * @param[out] output_ptr                            Pointer to the destination tensor.
 *                                                   Supported data types: QASYMM8
 * @param[in]  output_stride_x                       Stride of the destination tensor in X dimension
 *                                                   (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X
 *                                                   processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination tensor in Y dimension
 *                                                   (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y
 *                                                   processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the destination tensor in Z dimension
 *                                                   (in bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z
 *                                                   processed per workitem(in bytes)
 */
__kernel void binary_logical_op(TENSOR3D_DECLARATION(input1), TENSOR3D_DECLARATION(input2),
                                TENSOR3D_DECLARATION(output))
{
  Tensor3D input1 = CONVERT_TO_TENSOR3D_STRUCT(input1);
  Tensor3D input2 = CONVERT_TO_TENSOR3D_STRUCT(input2);
  Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

#if OP_CODE == 1 // LOGICAL AND
  VSTORE(VEC_SIZE)
  (CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input1.ptr) &&
             VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input2.ptr),
           VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)),
   0, (__global DATA_TYPE *)output.ptr);

#elif OP_CODE == 2 // LOGICAL OR
  VSTORE(VEC_SIZE)
  (CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input1.ptr) ||
             VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input2.ptr),
           VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)),
   0, (__global DATA_TYPE *)output.ptr);

#else // OP NOT SUPPORTED
  return

#endif
}
#endif // if defined(OP_CODE) && defined(DATA_TYPE)
