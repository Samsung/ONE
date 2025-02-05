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

#include "arm_compute/runtime/NEON/functions/NEInstanceNormalizationLayerEx.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
NEInstanceNormalizationLayerEx::NEInstanceNormalizationLayerEx(
  std::shared_ptr<IMemoryManager> memory_manager)
  : _memory_group(std::move(memory_manager)), _normalization_kernel(), _is_nchw(false),
    _permute_input(), _permute_output(), _permuted_input(), _permuted_output()
{
}

void NEInstanceNormalizationLayerEx::configure(ITensor *input, ITensor *output, ITensor *gamma,
                                               ITensor *beta, float epsilon)
{
  const DataLayout data_layout = input->info()->data_layout();

  // Configure Kernels
  _is_nchw = data_layout == DataLayout::NCHW;

  if (!_is_nchw)
  {
    _memory_group.manage(&_permuted_input);
    _memory_group.manage(&_permuted_output);

    // Configure the function to transform the input tensor from NHWC -> NCHW
    _permute_input.configure(input, &_permuted_input, PermutationVector(1U, 2U, 0U));
    _permuted_input.info()->set_data_layout(DataLayout::NCHW);

    _normalization_kernel.configure(&_permuted_input, &_permuted_output, gamma, beta, epsilon);
    _permuted_output.info()->set_data_layout(DataLayout::NCHW);

    _permute_output.configure(&_permuted_output, output != nullptr ? output : input,
                              PermutationVector(2U, 0U, 1U));
    _permuted_input.allocator()->allocate();
    _permuted_output.allocator()->allocate();
  }
  else
  {
    _normalization_kernel.configure(input, output, gamma, beta, epsilon);
  }
}

Status NEInstanceNormalizationLayerEx::validate(const ITensorInfo *input, const ITensorInfo *output,
                                                const ITensorInfo *gamma, const ITensorInfo *beta,
                                                float epsilon)
{
  return NEInstanceNormalizationLayerKernelEx::validate(
    &input->clone()->set_data_layout(DataLayout::NCHW),
    &output->clone()->set_data_layout(DataLayout::NCHW), gamma, beta, epsilon);
}

void NEInstanceNormalizationLayerEx::run()
{
  MemoryGroupResourceScope scope_mg(_memory_group);

  // Permute input
  if (!_is_nchw)
  {
    _permute_input.run();
  }

  NEScheduler::get().schedule(&_normalization_kernel, Window::DimZ);

  // Permute output
  if (!_is_nchw)
  {
    _permute_output.run();
  }
}
} // namespace arm_compute
