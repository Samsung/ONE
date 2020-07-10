/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * Copyright (c) 2016-2018 ARM Limited.
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

#include "arm_compute/core/CL/kernels/CLNegKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "support/StringSupport.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S16, DataType::S32,
                                                DataType::F16, DataType::F32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S16, DataType::S32,
                                                DataType::F16, DataType::F32);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(input->tensor_shape(), output->tensor_shape());
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

  return Status{};
}

} // namespace

CLNegKernel::CLNegKernel() : _input(nullptr), _output(nullptr) {}

void CLNegKernel::configure(const ICLTensor *input, ICLTensor *output)
{

  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

  _input = input;
  _output = output;

  constexpr unsigned int num_elems_processed_per_iteration = 16;

  // Create kernel
  std::set<std::string> build_opts;
  build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
  build_opts.emplace(
      ("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration)));
  _kernel =
      static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel("neg_tensor", build_opts));

  // Configure window
  Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

  AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
  AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
  update_window_and_padding(win, input_access, output_access);
  output_access.set_valid_region(win, input->info()->valid_region());

  ICLKernel::configure_internal(win);
}

void CLNegKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

  Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
  Window slice = collapsed.first_slice_window_3D();

  do
  {
    unsigned int idx = 0;
    add_3D_tensor_argument(idx, _input, slice);
    add_3D_tensor_argument(idx, _output, slice);
    enqueue(queue, *this, slice, lws_hint());
  } while (collapsed.slide_window_slice_3D(slice));
}
