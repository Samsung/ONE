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
 * Copyright (c) 2017-2018 ARM Limited.
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

#ifndef __ARM_COMPUTE_CLSCALEFACTORSYMM8KERNEL_H__
#define __ARM_COMPUTE_CLSCALEFACTORSYMM8KERNEL_H__

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to perform min max search on a 3D tensor.
 */
class CLScaleFactorSymm8Kernel : public ICLKernel
{
public:
  /** Default constructor */
  CLScaleFactorSymm8Kernel();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLScaleFactorSymm8Kernel(const CLScaleFactorSymm8Kernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLScaleFactorSymm8Kernel &operator=(const CLScaleFactorSymm8Kernel &) = delete;
  /** Allow instances of this class to be moved */
  CLScaleFactorSymm8Kernel(CLScaleFactorSymm8Kernel &&) = default;
  /** Allow instances of this class to be moved */
  CLScaleFactorSymm8Kernel &operator=(CLScaleFactorSymm8Kernel &&) = default;
  /** Initialise the kernel's input and output.
   *
   * @param[in]  input  Input tensor with 2 dimensions. The first dimension will be interpreted as
   * batches. Data types supported: F32.
   * @param[out] output Output tensor with shape [batches] which stores the scale values for each 2D
   * input tensor.
   *                    The dimensions over the first must match the batched dimensions of the input
   * tensor. Data types supported: F32.
   */
  void configure(const ICLTensor *input, ICLTensor *output);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLScaleFactorSymm8Kernel
   *
   * @param[in] input  Input tensor info.  Data types supported: F32.
   * @param[in] output Output tensor info with shape [batches] which stores the scale values for
   * each 2D input tensor.
   *                   The dimensions over the first must match the batched dimensions of the input
   * tensor. Data types supported: F32.
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output);

  /** Resets global minimum and maximum
   *
   * @param[in,out] queue Command queue on which to map and unmap the min_max tensor
   */
  void reset(cl::CommandQueue &queue);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input;
  ICLTensor *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLSCALEFACTORSYMM8KERNEL_H__ */
