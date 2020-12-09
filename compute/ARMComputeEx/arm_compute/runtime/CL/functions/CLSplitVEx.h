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
#ifndef __ARM_COMPUTE_CLSPLITVEX__
#define __ARM_COMPUTE_CLSPLITVEX__

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/CL/functions/CLSlice.h"
#include "arm_compute/core/Types.h"
#include <vector>
#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLSplitVKernel */
class CLSplitVEx : public IFunction
{
public:
  /** Default constructor */
  CLSplitVEx();
  /** Configure the split CL kernel
   *
   * @param[in]  input       The input tensor to split. Data types supported:
   * U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
   * @param[in]  size_splits A 1-D tensor containing the number of tensor values per split
   * @param[out] outputs     A vector containing the output tensor. Data types supported: Same as @p
   * input
   *                         The output tensors should match the input tensor dimensions for all
   * shape dimensions apart
   *                         from the split dimension.
   * @param[in]  split_dim   Integer value representing the input tensor dimension along which to
   * split
   * @param[in]  num_splits  Number of splits
   */
  void configure(const ICLTensor *input, const ICLTensor *size_splits, uint32_t split_dim,
                 const std::vector<ICLTensor *> &outputs, unsigned int num_splits);

  void run() override;

private:
  const ICLTensor *_input;
  const ICLTensor *_size_splits;
  std::vector<ICLTensor *> _outputs;
  unsigned int _num_splits;
  std::vector<CLSlice> _slice_functions;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLSPLITVEX__ */
