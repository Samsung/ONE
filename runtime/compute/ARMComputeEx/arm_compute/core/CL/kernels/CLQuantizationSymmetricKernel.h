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

#ifndef __ARM_COMPUTE_CLQUANTIZATIONSYMMETRICKERNEL_H__
#define __ARM_COMPUTE_CLQUANTIZATIONSYMMETRICKERNEL_H__

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the quantization layer kernel.
 *
 * @note The implementation supports only 2D input tensors.
 */
class CLQuantizationSymmetricKernel : public ICLKernel
{
public:
  /** Default constructor */
  CLQuantizationSymmetricKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLQuantizationSymmetricKernel(const CLQuantizationSymmetricKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CLQuantizationSymmetricKernel &operator=(const CLQuantizationSymmetricKernel &) = delete;
  /** Default Move Constructor. */
  CLQuantizationSymmetricKernel(CLQuantizationSymmetricKernel &&) = default;
  /** Default move assignment operator */
  CLQuantizationSymmetricKernel &operator=(CLQuantizationSymmetricKernel &&) = default;
  /** Default destructor */
  ~CLQuantizationSymmetricKernel() = default;
  /** Set the input, output.
   *
   * @param[in]  input  Source tensor. Data types supported: F32/F16.
   * @param[in] scale_factor Scale tensor of @p output. Data type supported: Same as @p input.
   * @param[out] output Destination tensor with the same dimensions of input. Data types supported:
   * S8.
   *
   * @note Output auto initialization is not supported by this kernel
   */
  void configure(const ICLTensor *input, const ICLTensor *scale_factor, ICLTensor *output);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * CLQuantizationSymmetricKernel
   *
   * @param[in] input  Input tensor info. Data types supported: F32/F16.
   * @param[in] scale_factor Scale tensor of @p output. Data type supported: Same as @p input.
   * @param[in] output Destination tensor info with the same dimensions of input. Data types
   * supported: S8.
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *scale_factor,
                         const ITensorInfo *output);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

private:
  const ICLTensor *_input;
  const ICLTensor *_scale_factor;
  ICLTensor *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLQUANTIZATIONSYMMETRICKERNEL_H__ */
