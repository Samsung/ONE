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

#ifndef __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCOREEX_H__
#define __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCOREEX_H__

#include "arm_compute/core/CL/kernels/CLDepthConvertLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyKernelEx.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpReductionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMReshapeRHSMatrixKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
class IMemoryManager;
class ICLTensor;

/** Basic function to execute GEMMLowpMatrixMultiplyCore on OpenCL. This function calls the
 * following OpenCL kernels:
 *
 *  -# @ref CLGEMMLowpMatrixMultiplyKernel (if the parameter "reshape_b_only_on_first_run" of
 * GEMMInfo is FALSE)
 *  -# @ref CLGEMMLowpMatrixAReductionKernel (if the offset of matrix B is not 0)
 *  -# @ref CLGEMMLowpMatrixBReductionKernel (if the offset of matrix A is not 0)
 *
*/
class CLGEMMLowpMatrixMultiplyCoreEx : public IFunction
{
public:
  /** Constructor */
  CLGEMMLowpMatrixMultiplyCoreEx(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLGEMMLowpMatrixMultiplyCoreEx(const CLGEMMLowpMatrixMultiplyCoreEx &) = delete;
  /** Default move constructor */
  CLGEMMLowpMatrixMultiplyCoreEx(CLGEMMLowpMatrixMultiplyCoreEx &&) = default;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLGEMMLowpMatrixMultiplyCoreEx &operator=(const CLGEMMLowpMatrixMultiplyCoreEx &) = delete;
  /** Default move assignment operator */
  CLGEMMLowpMatrixMultiplyCoreEx &operator=(CLGEMMLowpMatrixMultiplyCoreEx &&) = default;
  /** Initialise the kernel's inputs, output
   *
   * @note GEMMLowp:  low precision GEMM kernel. [A * B + C]
   *  This kernel performs the following computations:
   *
   *  -# Convert a values from QASYMM8 to int32 and add a_offset to each of them.
   *  -# Convert b values from QASYMM8 to int32 and add b_offset to each of them.
   *  -# Compute the matrix product of the resulting a * b in int32.
   *  -# Quantize to uint8 if gemm_info.gemmlowp_output_stage != NONE
   *
   * @param[in]  a         First input tensor  (Matrix A). Data type supported: QASYMM8.
   * @param[in]  b         Second input tensor (Matrix B). Data type supported: same as @p a
   * @param[in]  c         Third input tensor  (Matrix C). It can be a nullptr. Data type supported:
   * S32
   * @param[out] output    Output tensor. Data type supported: S32 or QASYMM8 if
   * gemm_info.gemmlowp_output_stage != NONE
   * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped
   * and
   *                       if the reshape of matrix B should be executed only for the first run
   */
  void configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output,
                 const GEMMInfo &gemm_info = GEMMInfo());
  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLGEMMLowpMatrixMultiplyCoreEx
   *
   * @param[in] a         First input tensor info (Matrix A). Data type supported: QASYMM8.
   * @param[in] b         Second input tensor info (Matrix B). Data type supported: same as @p a
   * @param[in] c         Third input tensor info (Matrix C). It can be a nullptr. Data type
   * supported: S32
   * @param[in] output    Output tensor info. Data type supported: S32 or QASYMM8 if
   * gemm_info.gemmlowp_output_stage != NONE
   * @param[in] gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped
   * and
   *                      if the reshape of matrix B should be executed only for the first run
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c,
                         const ITensorInfo *output, const GEMMInfo &gemm_info = GEMMInfo());

  // Inherited methods overridden:
  void run() override;
  void prepare() override;

private:
  MemoryGroup _memory_group;

  // Kernels used
  CLGEMMLowpMatrixMultiplyKernelEx _mm_midgard_kernel;
  CLGEMMLowpMatrixAReductionKernel _mtx_a_reduction_kernel;
  CLGEMMLowpMatrixBReductionKernel _mtx_b_reduction_kernel;

  // Temporary tensors
  CLTensor _vector_sum_col;
  CLTensor _vector_sum_row;

  int32_t _a_offset;
  int32_t _b_offset;
  bool _reshape_b_only_on_first_run;
  bool _is_prepared;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCOREEX_H__ */
