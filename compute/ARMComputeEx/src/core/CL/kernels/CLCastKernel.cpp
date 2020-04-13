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

#include "arm_compute/core/CL/kernels/CLCastKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"

using namespace arm_compute;

CLCastKernel::CLCastKernel() : _input(nullptr), _output(nullptr) {}

void CLCastKernel::configure(const ICLTensor *input, ICLTensor *output, SubDataType input_subtype)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::QASYMM8,
                                                DataType::S16, DataType::S32, DataType::F16,
                                                DataType::F32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::QASYMM8,
                                                DataType::S16, DataType::S32, DataType::F16,
                                                DataType::F32);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);

  _input = input;
  _output = output;

  constexpr unsigned int num_elems_processed_per_iteration = 16;

  // Set kernel build options
  CLBuildOptions build_opts;
  build_opts.add_option("-DDATA_TYPE_IN=" + get_cl_type_from_data_type(input->info()->data_type()));
  build_opts.add_option("-DDATA_TYPE_OUT=" +
                        get_cl_type_from_data_type(output->info()->data_type()));
  build_opts.add_option(
      ("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration)));

  // Create kernel
  if (is_data_type_quantized_asymmetric(input->info()->data_type()))
  {
    UniformQuantizationInfo qinfo = input->info()->quantization_info().uniform();
    const float scale_in = qinfo.scale;
    const int offset_in = qinfo.offset;
    build_opts.add_option("-DSCALE=" + float_to_string_with_full_precision(scale_in));
    build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(offset_in));

    _kernel = static_cast<cl::Kernel>(
        CLKernelLibraryEx::get().create_kernel("cast_qasymm_in", build_opts.options()));
  }
  else if (is_data_type_quantized_asymmetric(output->info()->data_type()))
  {
    UniformQuantizationInfo qinfo = output->info()->quantization_info().uniform();
    const float scale_in = qinfo.scale;
    const float offset_in = qinfo.offset;

    build_opts.add_option("-DSCALE=" + float_to_string_with_full_precision(scale_in));
    build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(offset_in));

    _kernel = static_cast<cl::Kernel>(
        CLKernelLibraryEx::get().create_kernel("cast_qasymm_out", build_opts.options()));
  }
  else
  {
    build_opts.add_option_if(input_subtype == SubDataType::BOOL, "-DBOOL_INPUT");
    _kernel = static_cast<cl::Kernel>(
        CLKernelLibraryEx::get().create_kernel("cast", build_opts.options()));
  }

  // Configure kernel window
  Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
  AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
  AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
  update_window_and_padding(win, input_access, output_access);
  output_access.set_valid_region(win, input->info()->valid_region());

  ICLKernel::configure_internal(win);
}

void CLCastKernel::run(const Window &window, cl::CommandQueue &queue)
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
