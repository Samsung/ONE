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

#include "arm_compute/core/CPP/kernels/CPPOneHotKernelEx.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/Traits.h"

namespace arm_compute
{
CPPOneHotKernelEx::CPPOneHotKernelEx()
    : _indices(nullptr), _output(nullptr), _depth(0), _on_value(0), _off_value(0), _axis(-1)
{
}

void CPPOneHotKernelEx::configure(const ITensor *indices, ITensor *output, const int depth,
                                  const float on_value, const float off_value, const int axis)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(indices, output);
  ARM_COMPUTE_ERROR_THROW_ON(validate(indices, depth, on_value, off_value, axis));

  _indices = indices;
  _output = output;
  _depth = depth;
  _on_value = on_value;
  _off_value = off_value;
  _axis = axis;

  ICPPKernel::configure(Window()); // Default 1 iteration window
}

Status CPPOneHotKernelEx::validate(const ITensor *indices, const int depth, const float on_value,
                                   const float off_value, const int axis)
{
  ARM_COMPUTE_UNUSED(on_value, off_value);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(indices, DataType::S32);
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(indices->info()->num_dimensions() != 1,
                                  "Only 1D indices are supported.");
  ARM_COMPUTE_RETURN_ERROR_ON(depth <= 0);
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis != -1, "Only axis = -1 is supported.");
  return Status{};
}

bool CPPOneHotKernelEx::is_parallelisable() const { return false; }

void CPPOneHotKernelEx::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IKernel::window(), window);

  const auto num_indices = _indices->info()->dimension(0);
  for (size_t i = 0; i < num_indices; ++i)
  {
    const auto index = *reinterpret_cast<int32_t *>(_indices->ptr_to_element(Coordinates{i}));
    for (int d = 0; d < _depth; ++d)
      *reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(d, i))) =
          (d == index) ? _on_value : _off_value;
  }
}
} // namespace arm_compute
