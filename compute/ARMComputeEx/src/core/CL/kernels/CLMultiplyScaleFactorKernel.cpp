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

#include "arm_compute/core/CL/kernels/CLMultiplyScaleFactorKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/StringSupport.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *scale_factor,
                          const ITensorInfo *output)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 2);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S32);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(scale_factor, 1, DataType::F16,
                                                       DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON(scale_factor->tensor_shape().total_size() == 0);
  ARM_COMPUTE_RETURN_ERROR_ON(scale_factor->num_dimensions() > 1);
  ARM_COMPUTE_RETURN_ERROR_ON(scale_factor->dimension(0) != input->dimension(1));
  ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape().total_size() == 0);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F16, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);

  // Checks performed when output is configured
  if ((output->total_size() != 0))
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
  }

  return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(const ITensorInfo *input,
                                                         ITensorInfo *output)
{
  // Configure kernel window
  Window win = calculate_max_window(*input, Steps());

  // Output tensor auto initialization if not yet initialized
  auto_init_if_empty(*output, input->tensor_shape(), 1, DataType::F32);

  // CLMultiplyScaleFactorKernel doesn't need padding so update_window_and_padding() can be
  // skipped
  Coordinates coord;
  coord.set_num_dimensions(output->num_dimensions());
  output->set_valid_region(ValidRegion(coord, output->tensor_shape()));

  return std::make_tuple(Status{}, win);
}
} // namespace

CLMultiplyScaleFactorKernel::CLMultiplyScaleFactorKernel()
  : _input(nullptr), _scale_factor(nullptr), _output(nullptr), _multiplier(1.f)
{
}

void CLMultiplyScaleFactorKernel::configure(const ICLTensor *input, const ICLTensor *scale_factor,
                                            ICLTensor *output, float multiplier)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_ERROR_THROW_ON(
    validate_arguments(input->info(), scale_factor->info(), output->info()));

  _input = input;
  _scale_factor = scale_factor;
  _output = output;
  _multiplier = multiplier;

  const int vec_size_x = 16 / output->info()->element_size();
  const int output_width_x = output->info()->tensor_shape().x();
  const bool multi_access_x = (output_width_x / vec_size_x > 0);

  // Create and update the window (if needed)
  Window win = calculate_max_window(*output->info());
  if (multi_access_x)
  {
    win.set(
      Window::DimX,
      Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), vec_size_x), vec_size_x));
  }
  ICLKernel::configure_internal(win);

  // Create kernel
  CLBuildOptions build_opts;
  build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
  build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(output->info()->data_type()));
  build_opts.add_option_if(
    multi_access_x, "-DLAST_ACCESSED_X=" +
                      support::cpp11::to_string(std::max<int>(output_width_x - vec_size_x, 0)));

  _kernel = static_cast<cl::Kernel>(
    CLKernelLibraryEx::get().create_kernel("multiply_scale_factor", build_opts.options()));
}

Status CLMultiplyScaleFactorKernel::validate(const ITensorInfo *input,
                                             const ITensorInfo *scale_factor,
                                             const ITensorInfo *output)
{
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, scale_factor, output));
  ARM_COMPUTE_RETURN_ON_ERROR(
    std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get())));
  return Status{};
}

void CLMultiplyScaleFactorKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

  Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
  Window slice = window_collapsed.first_slice_window_2D();

  // Set scale_factor window
  Window win_scale = calculate_max_window(*_scale_factor->info(), Steps());

  do
  {
    unsigned int idx = 0;
    add_2D_tensor_argument(idx, _input, slice);
    add_1D_tensor_argument(idx, _scale_factor, win_scale);
    add_2D_tensor_argument(idx, _output, slice);
    _kernel.setArg<float>(idx++, _multiplier);
    enqueue(queue, *this, slice, lws_hint());
  } while (window_collapsed.slide_window_slice_2D(slice));
}
