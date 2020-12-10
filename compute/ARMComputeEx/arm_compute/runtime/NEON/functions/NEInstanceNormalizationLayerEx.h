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
 * Copyright (c) 2019 ARM Limited.
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

#ifndef __ARM_COMPUTE_NEINSTANCENORMALIZATIONLAYEREX_H__
#define __ARM_COMPUTE_NEINSTANCENORMALIZATIONLAYEREX_H__

#include "arm_compute/core/NEON/kernels/NEInstanceNormalizationLayerKernelEx.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class ITensor;

/** Basic function to perform a Instance normalization.
 *
 * This function runs the following kernels:
 * -# @ref NEInstanceNormalizationLayerKernelEx
 */
class NEInstanceNormalizationLayerEx : public IFunction
{
public:
  /** Constructor */
  NEInstanceNormalizationLayerEx(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
  /** Set the input and output tensors.
   *
   * @param[in, out] input   Source tensor. In case of @p output tensor = nullptr this tensor will
   * store the result of the normalization.
   *                         Data types supported: F16/F32. Data layout supported: NHWC, NCHW
   * @param[out]     output  Destination tensor. Data types and data layouts supported: same as @p
   * input.
   * @param[in]      gamma   (Optional) The scale scalar value applied to the normalized tensor.
   * Defaults to 1.0
   * @param[in]      beta    (Optional) The offset scalar value applied to the normalized tensor.
   * Defaults to 0.0
   * @param[in]      epsilon (Optional) Lower bound value for the normalization. Defaults to 1e-12
   */
  void configure(ITensor *input, ITensor *output, ITensor *gamma, ITensor *beta,
                 float epsilon = 1e-12f);

  /** Static function to check if given info will lead to a valid configuration of @ref
   * NEInstanceNormalizationLayer.
   *
   * @param[in] input   Source tensor info. Data types supported: F16/F32. Data layout supported:
   * NHWC, NCHW
   * @param[in] output  Destination tensor info. Data types and data layouts supported: same as @p
   * input.
   * @param[in] gamma   (Optional) The scale scalar value applied to the normalized tensor. Defaults
   * to 1.0
   * @param[in] beta    (Optional) The offset scalar value applied to the normalized tensor.
   * Defaults to 0.0
   * @param[in] epsilon (Optional) Lower bound value for the normalization. Defaults to 1e-12
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                         const ITensorInfo *gamma = nullptr, const ITensorInfo *beta = nullptr,
                         float epsilon = 1e-12f);

  // Inherited methods overridden:
  void run() override;

private:
  MemoryGroup _memory_group;
  NEInstanceNormalizationLayerKernelEx _normalization_kernel;
  bool _is_nchw;
  NEPermute _permute_input;
  NEPermute _permute_output;
  Tensor _permuted_input;
  Tensor _permuted_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEINSTANCENORMALIZATIONLAYEREX_H__ */
