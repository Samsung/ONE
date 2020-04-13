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

#include "arm_compute/runtime/CL/functions/CLTransposeConvLayerUpsample.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/core/CL/ICLTensor.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;

CLTransposeConvLayerUpsample::CLTransposeConvLayerUpsample() // NOLINT
    : _upsample(),
      _output(nullptr)
{
}

Status CLTransposeConvLayerUpsample::validate(const ITensorInfo *input, const ITensorInfo *output,
                                              const BorderSize &inner_border,
                                              const PadStrideInfo &info)
{
  return CLTransposeConvLayerUpsampleKernel::validate(input, output, inner_border, info);
}

void CLTransposeConvLayerUpsample::configure(ICLTensor *input, ICLTensor *output,
                                             const BorderSize &inner_border,
                                             const PadStrideInfo &info)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

  _output = output;
  _upsample.configure(input, _output, inner_border, info);
}

void CLTransposeConvLayerUpsample::run()
{
  _output->map(CLScheduler::get().queue(), true);
  if (is_data_type_quantized_asymmetric(_output->info()->data_type()))
  {
    const uint8_t quantized_zero = _output->info()->quantization_info().uniform().offset;
    std::fill_n(_output->buffer(), _output->info()->total_size(), quantized_zero);
  }
  else
  {
    memset(_output->buffer(), 0, _output->info()->total_size());
  }
  _output->unmap(CLScheduler::get().queue());

  CLScheduler::get().enqueue(_upsample, false);
}
