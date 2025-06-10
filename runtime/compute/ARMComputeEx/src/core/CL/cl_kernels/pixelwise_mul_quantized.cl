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

#include "helpers_asymm.h"

#ifdef SATURATE
#define CONVERT_OP_FLOAT_STR(x, type, round) (convert_##type##_sat##round(x))
#else /* SATURATE */
#define CONVERT_OP_FLOAT_STR(x, type, round) (convert_##type##round(x))
#endif /* SATURATE */
#define CONVERT_OP_FLOAT(x, type, round) CONVERT_OP_FLOAT_STR(x, type, round)

#if defined(RESULT_OFFSET) && defined(RESULT_MULT_INT) && defined(RESULT_SHIFT)
/** Performs a pixelwise multiplication used to quantize down the int32 accumulator values of
 *  GEMMLowp to QASYMM8
 *
 * The following computations will be performed by the kernel:
 *
 *  -# Add offset terms to inputs
 *  -# Multiply inputs
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result by result_mult_int
 *  -# Shift the int32 accumulator by result_shift
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 *
 * @attention The inputs and output data types need to be passed at compile time using
 *            -DDATA_TYPE_IN1, -DDATA_TYPE_IN2 and -DDATA_TYPE_OUT:
 *            e.g. -DDATA_TYPE_IN1=uchar -DDATA_TYPE_IN2=uchar -DDATA_TYPE_OUT=uchar
 * @attention The offset factor of inputs must be passed at compile time using -DIN1_OFFSET and
 *            -DIN2_OFFSET
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor
 *            must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULT_INT and
 *            -DRESULT_SHIFT
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types:
 *                                               U8
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in
 *                                               bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed
 *                                               per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in
 *                                               bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed
 *                                               per workitem(in bytes)
 * @param[in]  in1_stride_z                      Stride of the source image in Y dimension (in
 *                                               bytes)
 * @param[in]  in1_step_z                        in1_stride_z * number of elements along Y processed
 *                                               per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types:
 *                                               U8
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in
 *                                               bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed
 *                                               per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in
 *                                               bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed
 *                                               per workitem(in bytes)
 * @param[in]  in2_stride_z                      Stride of the source image in Y dimension (in
 *                                               bytes)
 * @param[in]  in2_step_z                        in2_stride_z * number of elements along Y processed
 *                                               per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data
 *                                               types: U8
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in
 *                                               bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed
 *                                               per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in
 *                                              bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed
 *                                               per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the destination image in Y dimension (in
 *                                               bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Y processed
 *                                               per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination
 *                                               image
 * @param[in]  scale                             Float scaling factor. Supported data types: F32
 */
__kernel void pixelwise_mul_qasymm8(TENSOR3D_DECLARATION(in1), TENSOR3D_DECLARATION(in2),
                                    TENSOR3D_DECLARATION(out), const float scale)
{
  // Get pixels pointer
  Tensor3D in1 = CONVERT_TO_TENSOR3D_STRUCT(in1);
  Tensor3D in2 = CONVERT_TO_TENSOR3D_STRUCT(in2);
  Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

  // Load data
  VEC_DATA_TYPE(int, 16)
  in1_data = CONVERT(vload16(0, (__global DATA_TYPE_IN1 *)in1.ptr), VEC_DATA_TYPE(int, 16));
  VEC_DATA_TYPE(int, 16)
  in2_data = CONVERT(vload16(0, (__global DATA_TYPE_IN2 *)in2.ptr), VEC_DATA_TYPE(int, 16));

  // Perform multiplication of two inputs
  VEC_DATA_TYPE(int, 16) in1_val = in1_data + (VEC_DATA_TYPE(int, 16))(IN1_OFFSET);
  VEC_DATA_TYPE(int, 16) in2_val = in2_data + (VEC_DATA_TYPE(int, 16))(IN2_OFFSET);
  VEC_DATA_TYPE(int, 16) out_val = in1_val * in2_val;

  // Multiply with a multiplier smaller than 1
  out_val =
    ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(out_val, RESULT_MULT_INT, RESULT_SHIFT, 16);
  out_val += (VEC_DATA_TYPE(int, 16))(RESULT_OFFSET);

  VEC_DATA_TYPE(uchar, 16) res = CONVERT(out_val, VEC_DATA_TYPE(uchar, 16));

  // TODO: Apply min-max BOUND to support fuse with relu.
  /*
  #if defined(MIN_BOUND)
      res = max(res, (uchar16)MIN_BOUND);
  #endif // defined(MIN_BOUND)
  #if defined(MAX_BOUND)
      res = min(res, (uchar16)MAX_BOUND);
  #endif // defined(MAX_BOUND)
  */

  // Store result
  VSTORE(16)(CONVERT(res, VEC_DATA_TYPE(DATA_TYPE_OUT, 16)), 0, (__global DATA_TYPE_OUT *)out.ptr);
}
#endif // defined(RESULT_OFFSET) && defined(RESULT_MULT_INT) && defined(RESULT_SHIFT)
