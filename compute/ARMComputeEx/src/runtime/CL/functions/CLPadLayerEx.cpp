/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLPadLayerEx.h"
#include "arm_compute/core/CL/kernels/CLPadLayerKernelEx.h"

namespace arm_compute
{
CLPadLayerEx::CLPadLayerEx()
  : _pad_kernel(std::make_unique<CLPadLayerKernelEx>()),
    _copy_kernel(std::make_unique<opencl::kernels::ClCopyKernel>()), _perform_pad(false)
{
}

void CLPadLayerEx::configure(ICLTensor *input, ICLTensor *output, const PaddingList &padding,
                             PixelValue constant_value, PaddingMode mode)
{
  configure(CLKernelLibrary::get().get_compile_context(), input, output, padding, constant_value,
            mode);
}

void CLPadLayerEx::configure(const CLCompileContext &compile_context, ICLTensor *input,
                             ICLTensor *output, const PaddingList &padding,
                             PixelValue constant_value, PaddingMode mode)
{
  ARM_COMPUTE_ERROR_THROW_ON(
    validate(input->info(), output->info(), padding, constant_value, mode));

  _perform_pad = std::any_of(padding.begin(), padding.end(),
                             [](PaddingInfo info) { return info.first > 0 || info.second > 0; });

  if (_perform_pad)
  {
    _pad_kernel->configure(compile_context, input, output, padding, constant_value, mode);
  }
  else
  {
    Window copy_window = Window();
    copy_window.use_tensor_dimensions(output->info()->tensor_shape());
    // Copy the input to the whole output if no padding is applied
    _copy_kernel->configure(compile_context, input->info(), output->info(), &copy_window);
  }
}
Status CLPadLayerEx::validate(const ITensorInfo *input, const ITensorInfo *output,
                              const PaddingList &padding, PixelValue constant_value,
                              PaddingMode mode)
{
  bool perform_pad = std::any_of(padding.begin(), padding.end(), [](PaddingInfo info) {
    return info.first > 0 || info.second > 0;
  });

  if (perform_pad)
  {
    ARM_COMPUTE_RETURN_ON_ERROR(
      CLPadLayerKernelEx::validate(input, output, padding, constant_value, mode));
  }
  else
  {
    ARM_COMPUTE_RETURN_ON_ERROR(opencl::kernels::ClCopyKernel::validate(input, output));
  }
  return Status{};
}
void CLPadLayerEx::run()
{
  if (_perform_pad)
  {
    CLScheduler::get().enqueue(*_pad_kernel);
  }
  else
  {
    CLScheduler::get().enqueue(*_copy_kernel);
  }
}
} // namespace arm_compute
