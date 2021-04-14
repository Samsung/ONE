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
 * Copyright (c) 2018-2019 ARM Limited.
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

#ifndef __ARM_COMPUTE_NEBINARYLOGICALOPERATIONKERNEL_H__
#define __ARM_COMPUTE_NEBINARYLOGICALOPERATIONKERNEL_H__

#include "arm_compute/core/TypesEx.h"

#include "src/core/cpu/kernels/CpuElementwiseKernel.h"

namespace arm_compute
{

class NEBinaryLogicalOperationKernel : public cpu::kernels::CpuComparisonKernel
{
public:
  const char *name() const override { return "NEBinaryLogicalOperationKernel"; }

  NEBinaryLogicalOperationKernel() = default;
  /** Default destructor */
  ~NEBinaryLogicalOperationKernel() = default;

  /** Static function to check if given info will lead to a valid configuration of @ref
   * NEBinaryLogicalOperationKernel
   *
   * @param[in] op     Binary logical operation to be executed.
   * @param[in] input1 First tensor input. Data types supported: QASYMM8/U8.
   * @param[in] input2 Second tensor input. Data types supported: Same as @p input1.
   * @param[in] output Output tensor. Data types supported: Same as @p input1.
   */
  void configure(BinaryLogicalOperation op, const ITensor *input1, const ITensor *input2,
                 ITensor *output);

  /** Static function to check if given info will lead to a valid configuration of @ref
   * NEBinaryLogicalOperationKernel
   *
   * @param[in] op     Binary logical operation to be executed.
   * @param[in] input1 First tensor input info. Data types supported: QASYMM8/U8.
   * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
   * @param[in] output Output tensor info. Data types supported: Same as @p input1.
   *
   * @return a Status
   */
  static Status validate(BinaryLogicalOperation op, const ITensorInfo *input1,
                         const ITensorInfo *input2, const ITensorInfo *output);

protected:
  // Inherited methods overridden:
  static Status validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2,
                                   const ITensorInfo &output);

  std::function<void(const ITensor *input1, const ITensor *input2, ITensor *output,
                     const Window &window)>
    _function;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEBINARYLOGICALOPERATIONKERNEL_H__ */
