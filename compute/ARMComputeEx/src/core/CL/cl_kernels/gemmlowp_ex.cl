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

#if defined(NUM_ELEMS_PROCESSED_PER_THREAD_X) && defined(NUM_ELEMS_PROCESSED_PER_THREAD_Y) && \
    defined(COLS_A)
#define VECTOR_CHAR VEC_DATA_TYPE(char, NUM_ELEMS_PROCESSED_PER_THREAD_X)
#define VECTOR_INT VEC_DATA_TYPE(int, NUM_ELEMS_PROCESSED_PER_THREAD_X)
#define VECTOR_FLOAT VEC_DATA_TYPE(float, NUM_ELEMS_PROCESSED_PER_THREAD_X)
/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B
 * (src1) in case both matrices have not beed reshaped
 *
 * @attention The number of matrix A columns needs to be passed at compile time using -DCOLS_A
 *
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following
 * information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D
 * tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data type:
 * QASYMM8
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in
 * bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X
 * processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in
 * bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y
 * processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source
 * matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data type:
 * same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in
 * bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X
 * processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in
 * bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y
 * processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source
 * matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data
 * type: S32
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension
 * (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X
 * processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension
 * (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y
 * processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination
 * matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in
 * bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in
 * bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension
 * (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for
 * the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements for
 * the output tensor (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemmlowp_mm_midgard_ex(IMAGE_DECLARATION(src0), IMAGE_DECLARATION(src1),
                                     IMAGE_DECLARATION(dst), uint src0_stride_z, uint src1_stride_z,
                                     uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                     ,
                                     uint src_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                     ,
                                     uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
)
{
  int idx = get_global_id(0) * NUM_ELEMS_PROCESSED_PER_THREAD_X;

  // Compute starting address for matrix A and Matrix B
  int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

  // Update address for the matrix A
  src_addr.s0 += get_global_id(1) * src0_stride_y * NUM_ELEMS_PROCESSED_PER_THREAD_Y;

  // Update address for the matrix B
  src_addr.s1 += idx;

#if defined(REINTERPRET_INPUT_AS_3D)
  // Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across
  // the z dimension
  // in order to take into account the presence of possible cross plane paddings
  //
  //  |                  |
  //  |      plane0      |
  //  |                  |
  //  |__________________|
  //  |******************|
  //  |  cross_plane_pad |
  //  |******************|
  //  |                  |
  //  |      plane1      |
  //  |                  |
  //  |__________________|

  // The plane (zin) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)
  // by HEIGHT_GEMM3D
  uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) /
              (uint4)HEIGHT_GEMM3D;
  zin = min(DEPTH_GEMM3D - 1, zin);

  // Add offset due to the cross plane paddings
  zin *= (src_cross_plane_pad * src0_stride_y);

  // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
  // multiply src0_stride_z by DEPTH_GEMM3D
  src_addr.s0 += get_global_id(2) * src0_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

  // Add offset for batched GEMM
  src_addr.s0 += get_global_id(2) * src0_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

#if defined(MATRIX_B_DEPTH)
  // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
  src_addr.s1 += (get_global_id(2) % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
  src_addr.s1 += get_global_id(2) * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

  int end_row_vec_a = src_addr.s0 + COLS_A;

  VECTOR_INT acc0 = 0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
  VECTOR_INT acc1 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
  VECTOR_INT acc2 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
  VECTOR_INT acc3 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
  VECTOR_INT acc4 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4

  for (; src_addr.s0 <= (end_row_vec_a - 2); src_addr += (int2)(2, 2 * src1_stride_y))
  {
    // Load values from matrix A
    char2 a0 = vload2(0, (__global char *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    char2 a1 = vload2(0, (__global char *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    char2 a2 = vload2(0, (__global char *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    char2 a3 = vload2(0, (__global char *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    char2 a4 = vload2(0, (__global char *)(src0_ptr + src_addr.s0 + 4 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    // Load values from matrix B
    VECTOR_CHAR b0 =
        VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, (__global char *)(src1_ptr + src_addr.s1));
    VECTOR_CHAR b1 = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(
        0, (__global char *)(src1_ptr + src_addr.s1 + src1_stride_y));

    // Accumulate
    acc0 += CONVERT(b0, VECTOR_INT) * (VECTOR_INT)a0.s0;
    acc0 += CONVERT(b1, VECTOR_INT) * (VECTOR_INT)a0.s1;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    acc1 += CONVERT(b0, VECTOR_INT) * (VECTOR_INT)a1.s0;
    acc1 += CONVERT(b1, VECTOR_INT) * (VECTOR_INT)a1.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    acc2 += CONVERT(b0, VECTOR_INT) * (VECTOR_INT)a2.s0;
    acc2 += CONVERT(b1, VECTOR_INT) * (VECTOR_INT)a2.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    acc3 += CONVERT(b0, VECTOR_INT) * (VECTOR_INT)a3.s0;
    acc3 += CONVERT(b1, VECTOR_INT) * (VECTOR_INT)a3.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    acc4 += CONVERT(b0, VECTOR_INT) * (VECTOR_INT)a4.s0;
    acc4 += CONVERT(b1, VECTOR_INT) * (VECTOR_INT)a4.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
  }

  for (; src_addr.s0 < end_row_vec_a; src_addr += (int2)(1, src1_stride_y))
  {
    // Load values from matrix A
    char a0 = *(__global char *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    char a1 = *(__global char *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    char a2 = *(__global char *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    char a3 = *(__global char *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    char a4 = *(__global char *)(src0_ptr + src_addr.s0 + 4 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    // Load values from matrix B
    VECTOR_CHAR b0 =
        VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, (__global char *)(src1_ptr + src_addr.s1));

    // Accumulate
    acc0 += CONVERT(b0, VECTOR_INT) * (VECTOR_INT)a0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    acc1 += CONVERT(b0, VECTOR_INT) * (VECTOR_INT)a1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    acc2 += CONVERT(b0, VECTOR_INT) * (VECTOR_INT)a2;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    acc3 += CONVERT(b0, VECTOR_INT) * (VECTOR_INT)a3;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    acc4 += CONVERT(b0, VECTOR_INT) * (VECTOR_INT)a4;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
  }

  const int z = get_global_id(2);

  // Compute destination address
  Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

#if defined(REINTERPRET_OUTPUT_AS_3D)
  // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across
  // the z dimension
  // in order to take into account the presence of possible cross plane paddings
  //
  //  |                  |
  //  |      plane0      |
  //  |                  |
  //  |__________________|
  //  |******************|
  //  |  cross_plane_pad |
  //  |******************|
  //  |                  |
  //  |      plane1      |
  //  |                  |
  //  |__________________|

  // The plane (zout) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)
  // by HEIGHT_GEMM3D
  uint8 zout = ((uint8)(0, 1, 2, 3, 4, 5, 6, 7) +
                (uint8)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) /
               (uint8)HEIGHT_GEMM3D;
  zout = min(DEPTH_GEMM3D - 1, zout);

  // Add offset due to the cross plane paddings
  zout *= (dst_cross_plane_pad * dst_stride_y);

  // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
  // multiply dst_stride_z by DEPTH_GEMM3D
  dst.ptr += z * dst_stride_z * DEPTH_GEMM3D;

  // Store the result
  VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
  (CONVERT(acc0, VECTOR_INT), 0, (__global int *)(dst.ptr + 0 * dst_stride_y + zout.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
  VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
  (CONVERT(acc1, VECTOR_INT), 0, (__global int *)(dst.ptr + 1 * dst_stride_y + zout.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
  VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
  (CONVERT(acc2, VECTOR_INT), 0, (__global int *)(dst.ptr + 2 * dst_stride_y + zout.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
  VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
  (CONVERT(acc3, VECTOR_INT), 0, (__global int *)(dst.ptr + 3 * dst_stride_y + zout.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
  VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
  (CONVERT(acc4, VECTOR_INT), 0, (__global int *)(dst.ptr + 4 * dst_stride_y + zout.s4));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4

#else // defined(REINTERPRET_OUTPUT_AS_3D)
  // Add offset for batched GEMM
  dst.ptr += z * dst_stride_z;

  // Store the result
  VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
  (CONVERT(acc0, VECTOR_INT), 0, (__global int *)(dst.ptr + 0 * dst_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
  VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
  (CONVERT(acc1, VECTOR_INT), 0, (__global int *)(dst.ptr + 1 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
  VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
  (CONVERT(acc2, VECTOR_INT), 0, (__global int *)(dst.ptr + 2 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
  VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
  (CONVERT(acc3, VECTOR_INT), 0, (__global int *)(dst.ptr + 3 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
  VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
  (CONVERT(acc4, VECTOR_INT), 0, (__global int *)(dst.ptr + 4 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}
#endif // defined(NUM_ELEMS_PROCESSED_PER_THREAD_X) && defined(NUM_ELEMS_PROCESSED_PER_THREAD_Y) &&
       // defined(COLS_A)
