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

#ifndef __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYKERNELEX_H__
#define __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYKERNELEX_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to multiply matrices
 *
 * @note This kernel should be used ONLY for Midgard architectures
 *
 * This kernel performs the following computation:
 *
 *  -# Convert a values from int8 to int32
 *  -# Convert b values from int8 to int32
 *  -# Compute the int32 matrix product of the resulting a * b and store the result as int32
 *
 */
class CLGEMMLowpMatrixMultiplyKernelEx : public ICLKernel
{
public:
  /** Default Constructor */
  CLGEMMLowpMatrixMultiplyKernelEx();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLGEMMLowpMatrixMultiplyKernelEx(const CLGEMMLowpMatrixMultiplyKernelEx &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLGEMMLowpMatrixMultiplyKernelEx &operator=(const CLGEMMLowpMatrixMultiplyKernelEx &) = delete;
  /** Allow instances of this class to be moved */
  CLGEMMLowpMatrixMultiplyKernelEx(CLGEMMLowpMatrixMultiplyKernelEx &&) = default;
  /** Allow instances of this class to be moved */
  CLGEMMLowpMatrixMultiplyKernelEx &operator=(CLGEMMLowpMatrixMultiplyKernelEx &&) = default;
  /** Initialise the kernel's input and output.
   *
   * @note This kernel should be used ONLY for Midgard architectures
   *
   * @param[in]  input0    Input tensor containing the LHS matrix. Data type supported: QASYMM8
   * @param[in]  input1    Input tensor containing the RHS matrix. Data type supported: same as @p
   * input0
   * @param[out] output    Output tensor to store the result of matrix multiplication. Data type
   * supported: S32
   * @param[in]  gemm_info (Optional) GEMM information used to retrieve the original dimensions of
   * the input matrices
   */
  void configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output,
                 const GEMMReshapeInfo &gemm_info = GEMMReshapeInfo());
  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLGEMMLowpMatrixMultiplyKernelEx
   *
   * @param[in] input0    Input tensor containing the LHS matrix. Data type supported: QASYMM8
   * @param[in] input1    Input tensor containing the RHS matrix. Data type supported: same as @p
   * input0
   * @param[in] output    Output tensor to store the result of matrix multiplication. Data type
   * supported: S32
   * @param[in] gemm_info (Optional) GEMM information used to retrieve the original dimensions of
   * the input matrices
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input0, const ITensorInfo *input1,
                         const ITensorInfo *output,
                         const GEMMReshapeInfo &gemm_info = GEMMReshapeInfo());

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input0;
  const ICLTensor *_input1;
  ICLTensor *_output;
  bool _slide_matrix_b;
  bool _reinterpret_input_as_3d;
  bool _reinterpret_output_as_3d;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYKERNELEX_H__*/
