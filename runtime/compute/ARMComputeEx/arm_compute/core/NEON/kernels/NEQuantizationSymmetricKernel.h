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

#ifndef __ARM_COMPUTE_NEQUANTIZATIONSYMMETRICKERNEL_H__
#define __ARM_COMPUTE_NEQUANTIZATIONSYMMETRICKERNEL_H__

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the dequantization layer kernel. */
class NEQuantizationSymmetricKernel : public INEKernel
{
public:
  const char *name() const override { return "NEQuantizationSymmetricKernel"; }
  /** Default constructor */
  NEQuantizationSymmetricKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  NEQuantizationSymmetricKernel(const NEQuantizationSymmetricKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  NEQuantizationSymmetricKernel &operator=(const NEQuantizationSymmetricKernel &) = delete;
  /** Default Move Constructor. */
  NEQuantizationSymmetricKernel(NEQuantizationSymmetricKernel &&) = default;
  /** Default move assignment operator */
  NEQuantizationSymmetricKernel &operator=(NEQuantizationSymmetricKernel &&) = default;
  /** Default destructor */
  ~NEQuantizationSymmetricKernel() = default;
  /** Set input, output tensors.
   *
   * @param[in]  input  Source tensor. Data type supported: F16/F32.
   * @param[out] output Destination tensor with the same dimensions of input. Data type supported:
   * S8.
   * @param[out] scale_factor Scale tensor of @p output. Data type supported: Same as @p input.
   */
  void configure(const ITensor *input, ITensor *output, ITensor *scale_factor);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * NEQuantizationSymmetricKernel
   *
   * @param[in] input  Input tensor info. Data types supported: F16/F32.
   * @param[in] output Output tensor info. Data types supported: S8.
   * @param[out] scale_factor Scale tensor of @p output. Data type supported: Same as @p input.
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                         const ITensorInfo *scale_factor);

  // Inherited methods overridden:
  void run(const Window &window, const ThreadInfo &info) override;

private:
  template <typename T> void quantize(const Window &window);

private:
  const ITensor *_input;
  ITensor *_output;
  ITensor *_scale_factor;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEQUANTIZATIONSYMMETRICKERNEL_H__ */
