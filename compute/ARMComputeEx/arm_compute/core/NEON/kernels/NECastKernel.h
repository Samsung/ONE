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

#ifndef __ARM_COMPUTE_NECASTKERNEL_H__
#define __ARM_COMPUTE_NECASTKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/TypesEx.h"

namespace arm_compute
{
class ITensor;

/** Interface for the cast layer kernel. */
class NECastKernel : public INEKernel
{
public:
  const char *name() const override { return "NECastKernel"; }
  /** Default constructor */
  NECastKernel();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  NECastKernel(const NECastKernel &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  NECastKernel &operator=(const NECastKernel &) = delete;
  /** Default Move Constructor. */
  NECastKernel(NECastKernel &&) = default;
  /** Default move assignment operator */
  NECastKernel &operator=(NECastKernel &&) = default;
  /** Default destructor */
  ~NECastKernel() = default;
  /** Set input, output tensors.
   *
   * @param[in]  input  Source tensor. Data type supported: U8/S8/QASYMM8/U32/S32/F32.
   * @param[out] output Destination tensor with the same dimensions of input. Data type supported:
   * U8/S8/QASYMM8/U32/S32/F32.
   * @param[in]  input_subtype  Sub data type of input.
   */
  void configure(const ITensor *input, ITensor *output, SubDataType input_subtype);
  /** Static function to check if given info will lead to a valid configuration of @ref NECastKernel
   *
   * @param[in] input  Input tensor info. Data types supported: U8/S8/QASYMM8/U32/S32/F32.
   * @param[in] output Output tensor info. Data types supported: U8/S8/QASYMM8/U32/S32/F32.
   * @param[in] input_subtype  Sub data type of input.
   *
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                         SubDataType input_subtype);

  // Inherited methods overridden:
  void run(const Window &window, const ThreadInfo &info) override;

private:
  const ITensor *_input;
  ITensor *_output;
  SubDataType _input_subtype;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NECASTKERNEL_H__ */
