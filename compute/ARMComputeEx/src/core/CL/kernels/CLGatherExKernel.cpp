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

#include "arm_compute/core/CL/kernels/CLGatherExKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/utils/misc/ShapeCalculatorEx.h"
#include "arm_compute/core/UtilsEx.h"

using namespace arm_compute;

namespace
{

inline Status validate_arguments(const ITensorInfo *input, const ITensorInfo *indices,
                                 const ITensorInfo *output, int axis)
{
  const uint32_t actual_axis = wrap_around(axis, static_cast<int>(input->num_dimensions()));
  ARM_COMPUTE_RETURN_ERROR_ON(indices->num_dimensions() > 3);
  ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
  ARM_COMPUTE_ERROR_ON(input->num_dimensions() + indices->num_dimensions() - 1 > 4);
  ARM_COMPUTE_RETURN_ERROR_ON(actual_axis >= input->num_dimensions());
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(
      input, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::U16, DataType::S16,
      DataType::U32, DataType::S32, DataType::F16, DataType::F32);

  if (output->total_size() != 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    TensorShape output_shape = arm_compute::misc::shape_calculator::compute_gather_shape_ex(
        input->tensor_shape(), indices->tensor_shape(), actual_axis);
    ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() != output->tensor_shape().total_size());
  }

  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32, DataType::S32);

  return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *indices,
                                                        ITensorInfo *output, int axis)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, indices);
  const uint32_t actual_axis = wrap_around(axis, static_cast<int>(input->num_dimensions()));
  std::unique_ptr<ITensorInfo> output_info = input->clone();
  output_info->set_tensor_shape(arm_compute::misc::shape_calculator::compute_gather_shape_ex(
      input->tensor_shape(), indices->tensor_shape(), actual_axis));
  // Output auto initialization if not yet initialized
  auto_init_if_empty((*output), output_info->tensor_shape(), 1, input->data_type());

  // Create window
  Window win = calculate_max_window(*output, Steps());
  output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

  return std::make_pair(Status{}, win);
}

} // namespace

CLGatherExKernel::CLGatherExKernel()
    : _input(nullptr), _indices(nullptr), _output(nullptr), _axis(0)
{
}

void CLGatherExKernel::configure(const ICLTensor *input, const ICLTensor *indices,
                                 ICLTensor *output, int axis)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, indices);
  ARM_COMPUTE_ERROR_THROW_ON(
      validate_arguments(input->info(), indices->info(), output->info(), axis));

  // Configure kernel window
  auto win_config =
      validate_and_configure_window(input->info(), indices->info(), output->info(), axis);
  ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

  _input = input;
  _output = output;
  _indices = indices;
  _axis = wrap_around(axis, static_cast<int>(input->info()->num_dimensions()));

  // Set build options
  CLBuildOptions build_opts;
  build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
  build_opts.add_option("-DOUTPUT_DIM_Z=" +
                        support::cpp11::to_string(output->info()->dimension(2)));
  build_opts.add_option("-DINPUT_DIM_Z=" + support::cpp11::to_string(input->info()->dimension(2)));
  build_opts.add_option("-DAXIS=" + support::cpp11::to_string(_axis));
  build_opts.add_option("-DINDICES_DIM=" +
                        support::cpp11::to_string(indices->info()->num_dimensions()));

  // Create kernel
  _kernel = static_cast<cl::Kernel>(
      CLKernelLibraryEx::get().create_kernel("gather_ex", build_opts.options()));
  ICLKernel::configure_internal(win_config.second);
}

Status CLGatherExKernel::validate(const ITensorInfo *input, const ITensorInfo *indices,
                                  const ITensorInfo *output, int axis)
{
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, indices, output, axis));
  ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(),
                                                            indices->clone().get(),
                                                            output->clone().get(), axis)
                                  .first);
  return Status{};
}

void CLGatherExKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ, 4);
  unsigned int idx = 0;
  add_4D_tensor_argument(idx, _input, window_collapsed);
  add_3D_tensor_argument(idx, _indices, window_collapsed);
  add_4D_tensor_argument(idx, _output, window_collapsed);
  enqueue(queue, *this, window_collapsed, lws_hint());
}
