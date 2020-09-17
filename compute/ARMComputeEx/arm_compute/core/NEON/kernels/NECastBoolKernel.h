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
 * Copyright (c) 2016-2020 ARM Limited.
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
#ifndef __ARM_COMPUTE_NECASTBOOLKERNEL_H__
#define __ARM_COMPUTE_NECASTBOOLKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/**
 * @brief Class for the kernel converting boolean type
 */
class NECastBoolKernel : public INEKernel
{
public:
  const char *name() const override { return "NECastBoolKernel"; }
  /** Default constructor*/
  NECastBoolKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  NECastBoolKernel(const NECastBoolKernel &) = delete;
  /** Default move constructor */
  NECastBoolKernel(NECastBoolKernel &&) = default;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  NECastBoolKernel &operator=(const NECastBoolKernel &) = delete;
  /** Default move assignment operator */
  NECastBoolKernel &operator=(NECastBoolKernel &&) = default;
  /** Set the input and output of the kernel
   *
   * Valid conversions Input -> Output :
   *
   *   - U8             -> U8, S8, U16, S16, U32, S32, F32, F16
   *
   * @param[in]  input  The input tensor to convert. Data types supported: U8
   * @param[out] output The output tensor. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
   */
  void configure(const ITensor *input, ITensor *output);
  /** Static function to check if given info will lead to a valid configuration of @ref
   * NECastBoolKernel
   *
   * @param[in] input  Source tensor info. Data types supported: U8
   * @param[in] output Destination tensor info. Data type supported: U8/S8/U16/S16/U32/S32/F16/F32.
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output);

  // Inherited methods overridden:
  void run(const Window &window, const ThreadInfo &info) override;

private:
  const ITensor *_input;
  ITensor *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NECASTBOOLKERNEL_H__ */
