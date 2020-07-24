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

#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCoreEx.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/gemm/reshaped_only_rhs/CLGEMMReshapedOnlyRHSKernelConfiguration.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::cl_gemm;

namespace
{
inline bool is_gemm_reshaped(bool reshape_b_only_on_first_run, GPUTarget gpu_target)
{
  return (get_arch_from_target(gpu_target) != GPUTarget::MIDGARD) && (reshape_b_only_on_first_run);
}
} // namespace

CLGEMMLowpMatrixMultiplyCoreEx::CLGEMMLowpMatrixMultiplyCoreEx(
    std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _mm_midgard_kernel(), _mtx_a_reduction_kernel(),
      _mtx_b_reduction_kernel(), _vector_sum_col(), _vector_sum_row(), _a_offset(0), _b_offset(0),
      _reshape_b_only_on_first_run(false), _is_prepared(false)
{
}

void CLGEMMLowpMatrixMultiplyCoreEx::configure(const ICLTensor *a, const ICLTensor *b,
                                               const ICLTensor *c, ICLTensor *output,
                                               const GEMMInfo &gemm_info)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
  ARM_COMPUTE_UNUSED(c);
  ARM_COMPUTE_ERROR_THROW_ON(CLGEMMLowpMatrixMultiplyCoreEx::validate(
      a->info(), b->info(), c != nullptr ? c->info() : nullptr, output->info(), gemm_info));

  _is_prepared = false;
  _reshape_b_only_on_first_run = gemm_info.reshape_b_only_on_first_run();
  _a_offset = a->info()->quantization_info().uniform().offset;
  _b_offset = b->info()->quantization_info().uniform().offset;

  // Get the GPU target
  const GPUTarget gpu_target = CLScheduler::get().target();

  // Set the target for the kernels
  _mm_midgard_kernel.set_target(gpu_target);

  // GEMMRHSMatrixInfo rhs_info;
  // GEMMLHSMatrixInfo lhs_info;

  // Arguments used by GEMMReshapeInfo
  // If we pass the matrix A and matrix B reshaped to CLGEMMMatrixMultiplyKernel, we need to pass m,
  // n, k, mult_transpose1xW_width and mult_interleave4x4_height to CLGEMMReshapeInfo
  // in order to know how the matrices have been reshaped
  bool reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
  const unsigned int m = reinterpret_input_as_3d
                             ? (a->info()->dimension(1) * a->info()->dimension(2))
                             : a->info()->dimension(1);
  const unsigned int n = b->info()->dimension(0);
  const unsigned int k = a->info()->dimension(0);
  const int depth_output_gemm3d = gemm_info.depth_output_gemm3d();

  const ICLTensor *matrix_b = b;
  // Configure matrix multiply kernel
  _mm_midgard_kernel.configure(
      a, matrix_b, output,
      GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d));
}

Status CLGEMMLowpMatrixMultiplyCoreEx::validate(const ITensorInfo *a, const ITensorInfo *b,
                                                const ITensorInfo *c, const ITensorInfo *output,
                                                const GEMMInfo &gemm_info)
{
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::S8);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
  ARM_COMPUTE_UNUSED(c);

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(),
                                  "Matrix A already reshaped is not supported");
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(),
                                  "Matrix B already reshaped is not supported");

  const ITensorInfo *matrix_a_info = a;

  // Get the GPU target
  const GPUTarget gpu_target = CLScheduler::get().target();

  bool reinterpret_input_as_3d = gemm_info.reinterpret_input_as_3d();
  const unsigned int m =
      reinterpret_input_as_3d ? (a->dimension(1) * a->dimension(2)) : a->dimension(1);
  const unsigned int n = b->dimension(0);
  const unsigned int k = a->dimension(0);
  const int depth_output_gemm3d = gemm_info.depth_output_gemm3d();

  bool reshape_matrix_b = is_gemm_reshaped(gemm_info.reshape_b_only_on_first_run(), gpu_target);

  const GEMMReshapeInfo reshape_info =
      GEMMReshapeInfo(m, n, k, 1, 1, depth_output_gemm3d, reinterpret_input_as_3d);

  TensorInfo weights_info(*b);
  const ITensorInfo *matrix_b_info = &weights_info;
  if (reshape_matrix_b)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(false,
                                    "CLGEMMLowpMatrixMultiplyCoreEx does not support reshape_b");
  }

  // Validate matrix multiply
  ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyKernelEx::validate(
      matrix_a_info, matrix_b_info, output, reshape_info));

  return Status{};
}

void CLGEMMLowpMatrixMultiplyCoreEx::run()
{
  prepare();

  MemoryGroupResourceScope scope_mg(_memory_group);

  // Run matrix multiply
  CLScheduler::get().enqueue(_mm_midgard_kernel, false);
}

void CLGEMMLowpMatrixMultiplyCoreEx::prepare()
{
  if (!_is_prepared)
  {
    _is_prepared = true;
  }
}
} // namespace arm_compute
