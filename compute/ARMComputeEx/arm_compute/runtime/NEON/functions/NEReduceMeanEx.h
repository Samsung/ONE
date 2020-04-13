/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * Copyright (c) 2018 ARM Limited.
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

#ifndef __ARM_COMPUTE_NEON_REDUCE_MEAN_EX_H__
#define __ARM_COMPUTE_NEON_REDUCE_MEAN_EX_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"

namespace arm_compute
{
class ITensor;

/** Basic function to perform reduce operation */
class NEReduceMeanEx : public IFunction
{
public:
  /** Constructor */
  NEReduceMeanEx(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
  /** Configure kernel
   *
   * @note Supported tensor rank: up to 4
   *
   * @param[in]  input          Source tensor. Data type supported: QASYMM8/F16/F32
   * @param[in]  reduction_axis Reduction axis vector.
   * @param[in]  keep_dims      If positive, retains reduced dimensions with length 1.
   * @param[out] output         Destination tensor. Data type supported: Same as @p input
   */
  void configure(ITensor *input, const Coordinates &reduction_axis, bool keep_dims,
                 ITensor *output);

  /** Static function to check if given info will lead to a valid configuration of @ref
   * NEReduceMeanEx
   *
   * @param[in] input          Source tensor. Data type supported: QASYMM8/F16/F32
   * @param[in] reduction_axis Reduction axis vector.
   * @param[in] keep_dims      If positive, retains reduced dimensions with length 1.
   * @param[in] output         Destination tensor. Data type supported: Same as @p input
   *
   * @return A status
   */
  static Status validate(const ITensorInfo *input, const Coordinates &reduction_axis,
                         bool keep_dims, const ITensorInfo *output);

  // Inherited methods overridden:
  void run() override;

private:
  MemoryGroup _memory_group;
  std::unique_ptr<NEReductionOperation[]> _reduction_kernels{nullptr};
  std::unique_ptr<Tensor[]> _reduced_outs{nullptr};
  NEReshapeLayer _reshape;
  unsigned int _reduction_ops;
  bool _keep_dims;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEON_REDUCE_MEAN_EX_H__ */
