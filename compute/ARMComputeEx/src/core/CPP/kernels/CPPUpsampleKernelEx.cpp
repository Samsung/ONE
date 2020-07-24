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

#include "arm_compute/core/CPP/kernels/CPPUpsampleKernelEx.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
CPPUpsampleKernelEx::CPPUpsampleKernelEx() : _input(nullptr), _output(nullptr), _info() {}

bool CPPUpsampleKernelEx::is_parallelisable() const { return false; }

void CPPUpsampleKernelEx::configure(const ITensor *input, ITensor *output,
                                    const PadStrideInfo &info)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

  _input = input;
  _output = output;
  _info = info;

  // Configure kernel window
  Window win = calculate_max_window(*input->info(), Steps());

  // The CPPUpsampleKernelEx doesn't need padding so update_window_and_padding() can be skipped
  Coordinates coord;
  coord.set_num_dimensions(output->info()->num_dimensions());
  output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

  ICPPKernel::configure(win);
}

void CPPUpsampleKernelEx::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

  // Initialize _scaled_output buffer
  const int width_scaled = _output->info()->dimension(0);
  const int height_scaled = _output->info()->dimension(1);
  const int stride_x = _info.stride().first;
  const int stride_y = _info.stride().second;
  const int start_x = _info.pad_left();
  const int start_y = _info.pad_top();
  const int end_y = height_scaled - _info.pad_bottom();
  const int end_x = width_scaled - _info.pad_top();
  const size_t element_size = _input->info()->element_size();

  // The fill value is normally 0, but for QASYMM8 the '0' corresponds to the offset
  const uint8_t fill_value =
      _output->info()->data_type() == DataType::QASYMM8
          ? utility::clamp<uint8_t>(_output->info()->quantization_info().uniform().offset)
          : 0;
  // Filling a value different than 0 works only for QASYMM8 datatype since we are filling 1byte
  // values in a buffer of uint8_ts
  std::fill_n(_output->buffer(), _output->info()->total_size(), fill_value);

  // Create window
  Window window_out(window);
  window_out.set(Window::DimX, Window::Dimension(start_x, end_x, stride_x));
  window_out.set(Window::DimY, Window::Dimension(start_y, end_y, stride_y));

  // Create iterators
  Iterator in(_input, window);
  Iterator out(_output, window_out);

  execute_window_loop(
      window, [&](const Coordinates &) { memcpy(out.ptr(), in.ptr(), element_size); }, in, out);
}
} // namespace arm_compute
