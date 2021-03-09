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

#ifndef __ARM_COMPUTE_CLBINARYLOGICALOPKERNEL_H__
#define __ARM_COMPUTE_CLBINARYLOGICALOPKERNEL_H__

#include "arm_compute/core/TypesEx.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to return truth values of two input tensors for Binary Logical Op*/
class CLBinaryLogicalOpKernel : public ICLKernel
{
public:
  /** Default constructor */
  CLBinaryLogicalOpKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers). */
  CLBinaryLogicalOpKernel(const CLBinaryLogicalOpKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers). */
  CLBinaryLogicalOpKernel &operator=(const CLBinaryLogicalOpKernel &) = delete;
  /** Allow instances of this class to be moved */
  CLBinaryLogicalOpKernel(CLBinaryLogicalOpKernel &&) = default;
  /** Allow instances of this class to be moved */
  CLBinaryLogicalOpKernel &operator=(CLBinaryLogicalOpKernel &&) = default;
  /** Initialize the kernel's input, output.
   *
   * @param[in]  input1  Source tensor1.
   * @param[in]  input2  Source tensor2.
   * @param[out] output  Output tensor.
   */
  void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output,
                 BinaryLogicalOperation op);

  // Inherited methods overridden:
  void run(const Window &window, cl::CommandQueue &queue) override;

  BorderSize border_size() const override;

private:
  const ICLTensor *_input1;
  const ICLTensor *_input2;
  ICLTensor *_output;
};

} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLBINARYLOGICALOPKERNEL_H__ */
