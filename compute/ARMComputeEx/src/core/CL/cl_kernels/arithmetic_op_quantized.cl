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
#define ADD(x, y) add_sat((x), (y))
#define SUB(x, y) sub_sat((x), (y))
#else /* SATURATE */
#define ADD(x, y) (x) + (y)
#define SUB(x, y) (x) - (y)
#endif /* SATURATE */

/** Performs a pixelwise addition used to quantize down the int32 accumulator values of GEMMLowp to
 *  QASYMM8
 *
 * The following computations will be performed:
 *
 *  -# Add offset terms to inputs
    -# Get scaled value of two inputs
 *  -# Add inputs
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result by result_mult_int
 *  -# Shift the int32 accumulator by result_shift
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 *
 * @attention The inputs and output data types need to be passed at compile time using
 *            -DDATA_TYPE_IN1, -DDATA_TYPE_IN2 and -DDATA_TYPE_OUT:
 *            e.g. -DDATA_TYPE_IN1=uchar -DDATA_TYPE_IN2=uchar -DDATA_TYPE_OUT=uchar
 * @attention The number of bits to shift left of input tensors must be passed at compile time using
 *            -DLEFT_SHIFT
 * @attention The offset, scalar scale factor and number of bits to shift right of input tensors
 *            must be passed at compile time using -DIN1_OFFSET, -RIN1_MULT_INT, -DIN1_SHIFT,
 -DIN2_OFFSET,
 *            -RIN2_MULT_INT and -DIN2_SHIFT
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor
 *            must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULT_INT and
 -DRESULT_SHIFT
 *
 * @attention The input and output data_types need to be passed at compile time using
 *            -DDATA_TYPE_IN1, -DDATA_TYPE_IN2 and -DDATA_TYPE_OUT:
 *            e.g. -DDATA_TYPE_IN1=uchar -DDATA_TYPE_IN2=uchar -DDATA_TYPE_OUT=uchar
 * @attention The inputs and output scale information of qasymm8 need to be passed at compile time
 *            using -DSCALE_IN1, -DSCALE_IN2 and -DSCALE_OUT:
 *            e.g. -DSCALE_IN1=1.f -DSCALE_IN2=1.f -DSCALE_OUT=2.f
 * @attention The inputs and output scale offset need to be passed at compile time using
 *            -DOFFSET_IN1, -DOFFSET_IN2 and -DOFFSET_OUT:
 *            e.g. -DOFFSET_IN1=0 -DOFFSET_IN2=0 -DOFFSET_OUT=0
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g.
 *            -DVEC_SIZE=16
 * @attention To perform saturating operation -DSATURATE has to be passed to the compiler otherwise
 *            wrapping policy will be used.
 *
 * @param[in]  in1_ptr                           Pointer to the source tensor.
 *                                               Supported data types: QASYMM8
 * @param[in]  in1_stride_x                      Stride of the source tensor in X dimension
 *                                               (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed
 *                                               per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source tensor in Y dimension
 *                                               (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed
 *                                               per workitem(in bytes)
 * @param[in]  in1_stride_z                      Stride of the source tensor in Z dimension
 *                                               (in bytes)
 * @param[in]  in1_step_z                        in1_stride_z * number of elements along Z processed
 *                                               per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source
 *                                               tensor
 * @param[in]  in2_ptr                           Pointer to the source tensor. Supported data types:
 *                                               QASYMM8
 * @param[in]  in2_stride_x                      Stride of the source tensor in X dimension
 *                                               (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed
 *                                               per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source tensor in Y dimension
 *                                               (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed
 *                                               per workitem(in bytes)
 * @param[in]  in2_stride_z                      Stride of the source tensor in Z dimension
 *                                               (in bytes)
 * @param[in]  in2_step_z                        in2_stride_z * number of elements along Z processed
 *                                               per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source
 *                                               tensor
 * @param[out] out_ptr                           Pointer to the destination tensor.
 *                                               Supported data types: QASYMM8
 * @param[in]  out_stride_x                      Stride of the destination tensor in X dimension
 *                                               (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed
 *                                               per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination tensor in Y dimension
 *                                               (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed
 *                                               per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the source tensor in Z dimension
 *                                               (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Z processed
 *                                               per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination
 *                                               tensor
 */
__kernel void arithmetic_add_qasymm8(TENSOR3D_DECLARATION(in1), TENSOR3D_DECLARATION(in2),
                                     TENSOR3D_DECLARATION(out))
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

  // Get scaled value of two inputs
  VEC_DATA_TYPE(int, 16) in1_val = in1_data + (VEC_DATA_TYPE(int, 16))(IN1_OFFSET);
  VEC_DATA_TYPE(int, 16) in2_val = in2_data + (VEC_DATA_TYPE(int, 16))(IN2_OFFSET);

  VEC_DATA_TYPE(int, 16)
  left_shift = (VEC_DATA_TYPE(int, 16))1 << (VEC_DATA_TYPE(int, 16))(LEFT_SHIFT);
  VEC_DATA_TYPE(int, 16) shifted_in1_val = in1_val * left_shift;
  VEC_DATA_TYPE(int, 16) shifted_in2_val = in2_val * left_shift;

  VEC_DATA_TYPE(int, 16)
  scaled_in1_val =
      ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(shifted_in1_val, IN1_MULT_INT, IN1_SHIFT, 16);
  VEC_DATA_TYPE(int, 16)
  scaled_in2_val =
      ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(shifted_in2_val, IN2_MULT_INT, IN2_SHIFT, 16);

  // Add inputs and multiply with a multiplier smaller than 1
  VEC_DATA_TYPE(int, 16) sum_val = scaled_in1_val + scaled_in2_val;
  VEC_DATA_TYPE(int, 16)
  out_val =
      ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(sum_val, RESULT_MULT_INT, RESULT_SHIFT, 16);
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
