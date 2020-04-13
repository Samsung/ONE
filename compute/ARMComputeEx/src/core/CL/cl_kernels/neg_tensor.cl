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

#if defined(DATA_TYPE)
/** Performs a negation of input tensor.
 *
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g.
 *            -DVEC_SIZE=16
 * @attention Data type can be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types:
 *                                               S16/S32/F16/F32.
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in
 *                                               bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed
 *                                               per work item (in bytes)
 * @param[in]  in_offset_first_element_in_bytes  Offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data
 *                                               types: same as @p input_ptr
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in
 *                                               bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed
 *                                               per work item (in bytes)
 * @param[in]  out_offset_first_element_in_bytes Offset of the first element in the destination
 * image
 *
 */
__kernel void neg_tensor(TENSOR3D_DECLARATION(input), TENSOR3D_DECLARATION(output))
{
  Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
  Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);

  VSTORE(VEC_SIZE)
  (-VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input.ptr), 0, (__global DATA_TYPE *)output.ptr);
}
#endif // defined(DATA_TYPE)
