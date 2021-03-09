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

#include "arm_compute/core/CL/kernels/CLScaleFactorSymm8Kernel.h"

#include "src/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/helpers/AutoConfiguration.h"

#include "support/StringSupport.h"

#include <climits>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 2);

  if (output->tensor_shape().total_size() > 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    TensorShape output_shape = TensorShape{input->dimension(1)};

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
  }

  return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
  TensorShape output_shape = TensorShape{input->dimension(1)};

  // Output auto initialization if not yet initialized
  auto_init_if_empty(*output, output_shape, 1, input->data_type());

  const unsigned int num_elems_processed_per_iteration = 1;

  // Configure kernel window
  Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
  AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
  AccessWindowStatic output_access(output, 0, 0, output->dimension(0), 1);

  bool window_changed = update_window_and_padding(win, input_access, output_access);

  output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

  Status err = (window_changed)
                 ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!")
                 : Status{};
  return std::make_tuple(err, win);
}
} // namespace

CLScaleFactorSymm8Kernel::CLScaleFactorSymm8Kernel() : _input(nullptr), _output(nullptr) {}

void CLScaleFactorSymm8Kernel::configure(const ICLTensor *input, ICLTensor *output)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

  _input = input;
  _output = output;

  std::set<std::string> build_opts;
  build_opts.emplace("-DWIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));

  // Create kernel
  _kernel = static_cast<cl::Kernel>(
    CLKernelLibraryEx::get().create_kernel("scale_factor_symm8", build_opts));

  auto win_config = validate_and_configure_window(input->info(), output->info());

  ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

  ICLKernel::configure_internal(std::get<1>(win_config));
}

Status CLScaleFactorSymm8Kernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
  ARM_COMPUTE_RETURN_ON_ERROR(
    std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get())));

  return Status{};
}

void CLScaleFactorSymm8Kernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
  Window slice = window_collapsed.first_slice_window_2D();
  slice.set(Window::DimX, Window::Dimension(0, 1, 1));

  do
  {
    Window output_slice = slice.shift_dimensions(1);

    unsigned int idx = 0;
    // Set inputs
    add_2D_tensor_argument(idx, _input, slice);
    add_1D_tensor_argument(idx, _output, output_slice);
    enqueue(queue, *this, slice, lws_hint());
  } while (window_collapsed.slide_window_slice_2D(slice));
}
