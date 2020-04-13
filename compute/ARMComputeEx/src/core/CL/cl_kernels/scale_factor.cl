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

#if defined(WIDTH)
/** This function identifies the min and maximum value of an input 3D tensor.
 *
 * @note The width, height and depth of the input tensor must be provided at compile time using
 * -DWIDTH, -DHEIGHT and -DDEPTH (e.g. -DWIDTH=320, -DHEIGHT=240, -DDEPTH=3)
 *
 * @param[in] src_ptr                           Pointer to the source tensor. Supported data types:
 * F32
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed
 * per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed
 * per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] dst_ptr                           Pointer to the min/max vector. Minimum value in
 * position 0, maximum value in position 1. Supported data types: F32.
 * @param[in] dst_stride_x                      Stride of the min/max vector in X dimension (in
 * bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed
 * per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the min/max
 * vector
 */
__kernel void scale_factor_symm8(IMAGE_DECLARATION(src), VECTOR_DECLARATION(dst))
{
  Image src = CONVERT_TO_IMAGE_STRUCT(src);

  float4 min_value = (float4)FLT_MAX;
  float4 max_value = (float4)-FLT_MAX;

  int x = 0;
  __global float *src_addr = (__global float *)(src.ptr);

  for (; x <= (int)(WIDTH - 8); x += 8)
  {
    float8 value = vload8(0, (__global float *)(src_addr + x));

    min_value = select(value.s0123, min_value, min_value < value.s0123);
    min_value = select(value.s4567, min_value, min_value < value.s4567);

    max_value = select(value.s0123, max_value, max_value > value.s0123);
    max_value = select(value.s4567, max_value, max_value > value.s4567);
  }

  for (; x < WIDTH; ++x)
  {
    float value = *(src_addr + x);

    min_value.s0 = min(min_value.s0, value);
    max_value.s0 = max(max_value.s0, value);
  }

  // Perform min/max reduction
  min_value.s01 = min(min_value.s01, min_value.s23);
  min_value.s0 = min(min_value.s0, min_value.s1);
  max_value.s01 = max(max_value.s01, max_value.s23);
  max_value.s0 = max(max_value.s0, max_value.s1);

  // Extract scale
  max_value.s0 = max(fabs(min_value.s0), fabs(max_value.s0)) / 127.0f;

  // Store min and max
  *((__global float *)(dst_ptr) + get_global_id(1)) = max_value.s0;
}
#endif // defined(WIDTH)
