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

#ifndef __ARM_COMPUTE_CPPONEHOTERNEL_H__
#define __ARM_COMPUTE_CPPONEHOTERNEL_H__

#include "arm_compute/core/CPP/ICPPKernel.h"

namespace arm_compute
{
class ITensor;

/** CPP kernel to perform tensor OneHot operation. */
class CPPOneHotKernelEx : public ICPPKernel
{
public:
  const char *name() const override { return "CPPOneHotKernelEx"; }
  /** Default constructor */
  CPPOneHotKernelEx();
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CPPOneHotKernelEx(const CPPOneHotKernelEx &) = delete;
  /** Prevent instances of this class from being copied (As this class contains pointers) */
  CPPOneHotKernelEx &operator=(const CPPOneHotKernelEx &) = delete;
  /** Allow instances of this class to be moved */
  CPPOneHotKernelEx(CPPOneHotKernelEx &&) = default;
  /** Allow instances of this class to be moved */
  CPPOneHotKernelEx &operator=(CPPOneHotKernelEx &&) = default;
  /** Default destructor */
  ~CPPOneHotKernelEx() = default;

  /** Set the input and output of the kernel.
   *
   * @param[in]  indices     A tensor for indices. Data types supported: S32
   * @param[in]  depth       A tensor for depth. Data types supported: S32
   * @param[in]  on_value    A tensor for on_value. Data types supported: F32
   * @param[in]  off_value   A tensor for off_value. Data types supported: F32*
   * @param[out] output      A tensor for computed value of one hot operator
   * @param[in]  axis        An int value for axis
   */
  void configure(const ITensor *indices, const ITensor *depth, const ITensor *on_value,
                 const ITensor *off_value, ITensor *output, const int axis);

  /** Static function to check if given info will lead to a valid configuration of @ref
   * CPPOneHotKernelEx
   *
   * @param[in]  indices     A tensor for indices. Data types supported: S32
   * @param[in]  depth       A tensor for depth. Data types supported: S32
   * @param[in]  on_value    A tensor for on_value. Data types supported: F32
   * @param[in]  off_value   A tensor for off_value. Data types supported: F32*
   * @param[in]  axis        An int value for axis
   *
   * @return a status
   */
  static Status validate(const ITensor *indices, const ITensor *depth, const ITensor *on_value,
                         const ITensor *off_value, const int axis);

  // Inherited methods overridden:
  void run(const Window &window, const ThreadInfo &info) override;
  bool is_parallelisable() const override;

private:
  /** Template function to run the topKV operation. */
  template <typename T> void run_one_hot();

  const ITensor *_indices;
  const ITensor *_depth;
  const ITensor *_on_value;
  const ITensor *_off_value;
  ITensor *_output;
  int _axis;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CPPONEHOTKERNEL_H__ */
