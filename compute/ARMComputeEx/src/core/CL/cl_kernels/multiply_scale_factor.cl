/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * Copyright (c) 2017-2019 ARM Limited.
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

#if defined(VEC_SIZE) && defined(DATA_TYPE)

/** This performs to multiply input by scale_factor.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g.
 * -DDATA_TYPE=float
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g.
 * -DVEC_SIZE=16
 * @note Quantization scale of input tensor is passed in with -DSCALE=scale.
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data
 * types: S8
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in
 * bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X
 * processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in
 * bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y
 * processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source
 * tensor
 * @param[in]  scale_ptr                            Pointer to the source tensor. Supported data
 * types: S32
 * @param[in]  scale_stride_x                       Stride of the source tensor in X dimension (in
 * bytes)
 * @param[in]  scale_step_x                         scale_stride_x * number of elements along X
 * processed per workitem(in bytes)
 * @param[in]  scale_offset_first_element_in_bytes  The offset of the first element in the scale
 * tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported
 * data types: F16/F32
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension
 * (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X
 * processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension
 * (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y
 * processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the
 * destination tensor
 */
__kernel void multiply_scale_factor(IMAGE_DECLARATION(input), VECTOR_DECLARATION(scale),
                                    IMAGE_DECLARATION(output), float multiplier)
{
  // Get pixels pointer
  Image input = CONVERT_TO_IMAGE_STRUCT(input);
  Image output = CONVERT_TO_IMAGE_STRUCT(output);

#if defined(VEC_SIZE) && defined(LAST_ACCESSED_X)
  // Check if access on width gets out of bounds
  // If it does shift access vector to access elements within bounds
  const int xi = (int)(get_global_id(0) * VEC_SIZE);
  input.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * input_stride_x;
  output.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * output_stride_x;

  // Load data
  VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
  val = CONVERT(VLOAD(VEC_SIZE)(0, (__global int *)input.ptr), VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));

  // Create scale vector
  VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
  vscale = *(((__global DATA_TYPE *)(scale_ptr)) + get_global_id(1));

  // Dequantize
  vscale *= (DATA_TYPE)(multiplier);
  val *= vscale;

  // Store result
  VSTORE(VEC_SIZE)
  (val, 0, (__global DATA_TYPE *)output.ptr);
#else  // !defined(VEC_SIZE) || !defined(LAST_ACCESSED_X)
  *((__global DATA_TYPE *)(output.ptr)) =
    ((DATA_TYPE)(*((__global int *)(input.ptr)))) *
    *(((__global DATA_TYPE *)(scale_ptr)) + get_global_id(1)) * (DATA_TYPE)(multiplier);
#endif // defined(VEC_SIZE) && defined(LAST_ACCESSED_X)
}

#endif // defined(VEC_SIZE) && defined(DATA_TYPE)
