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
 * Copyright (c) 2016-2018 ARM Limited.
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

#ifndef __ARM_COMPUTE_CLPRELU_KERNEL_H__
#define __ARM_COMPUTE_CLPRELU_KERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to calculate PReLU*/
class CLPReLUKernel : public ICLKernel
{
public:
  /** Default constructor */
  CLPReLUKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers). */
  CLPReLUKernel(const CLPReLUKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers). */
  CLPReLUKernel &operator=(const CLPReLUKernel &) = delete;
  /** Allow instances of this class to be moved */
  CLPReLUKernel(CLPReLUKernel &&) = default;
  /** Allow instances of this class to be moved */
  CLPReLUKernel &operator=(CLPReLUKernel &&) = default;
  /** Initialize the kernel's input, output.
   *
   * @param[in]  input  Source tensor1.
   * @param[in]  alpha  Source tensor2.
   * @param[out] output  Output tensor.
   */
  void configure(const ICLTensor *input, const ICLTensor *alpha, ICLTensor *output);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

  BorderSize border_size() const override;

private:
  const ICLTensor *_input;
  const ICLTensor *_alpha;
  ICLTensor *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLPRELU_KERNEL_H__ */
