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
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/core/CL/kernels/CLOneHotKernel.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/misc/ShapeCalculatorEx.h"
#include "support/StringSupport.h"
#include <string>
namespace arm_compute
{
namespace
{
inline Status validate_arguments(const ITensorInfo *indices, const ITensorInfo *on_value,
                                 const ITensorInfo *output, int depth, int axis)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(indices, on_value, output);
  const uint32_t actual_axis = wrap_around(axis, static_cast<int>(output->num_dimensions()));
  ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() > 4);
  ARM_COMPUTE_RETURN_ERROR_ON(on_value->tensor_shape().total_size() != 1);
  ARM_COMPUTE_RETURN_ERROR_ON(depth <= 0);
  ARM_COMPUTE_RETURN_ERROR_ON(actual_axis >= output->num_dimensions());
  ARM_COMPUTE_RETURN_ERROR_ON(on_value->data_type() == DataType::UNKNOWN);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(on_value, 1, DataType::U8, DataType::S8,
                                                       DataType::U16, DataType::S16, DataType::F16,
                                                       DataType::U32, DataType::S32, DataType::F32);
  if (output->total_size() != 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(on_value, output);
    TensorShape output_shape = arm_compute::misc::shape_calculator::compute_onehot_shape_ex(
        indices->tensor_shape(), static_cast<uint32_t>(depth), actual_axis);
    ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() != output->tensor_shape().total_size());
  }
  return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *indices,
                                                        const ITensorInfo *on_value,
                                                        ITensorInfo *output, int depth, int axis)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(indices, on_value, output, indices);
  const uint32_t actual_axis = wrap_around(axis, static_cast<int>(output->num_dimensions()));
  // Output auto initialization if not yet initialized
  TensorShape output_shape = arm_compute::misc::shape_calculator::compute_onehot_shape_ex(
      indices->tensor_shape(), static_cast<uint32_t>(depth), actual_axis);
  auto_init_if_empty((*output), output_shape, 1, on_value->data_type());
  // Create window
  Window win = calculate_max_window(*output, Steps());
  output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));
  return std::make_pair(Status{}, win);
}
} // namespace
CLOneHotKernel::CLOneHotKernel()
    : _indices(nullptr), _on_value(nullptr), _off_value(nullptr), _output(nullptr),
      _is_off_value_memset(false)
{
}
void CLOneHotKernel::configure(const ICLTensor *indices, const ICLTensor *on_value,
                               const ICLTensor *off_value, ICLTensor *output, int depth, int axis)
{
  _is_off_value_memset = false;
  ARM_COMPUTE_ERROR_ON_NULLPTR(indices, on_value, off_value, output);
  ARM_COMPUTE_ERROR_ON_NULLPTR(off_value->info());
  ARM_COMPUTE_ERROR_ON(off_value->info()->tensor_shape().total_size() != 1);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(on_value, off_value);
  _off_value = off_value;
  configure_common(indices, on_value, output, depth, axis);
}
void CLOneHotKernel::configure(const ICLTensor *indices, const ICLTensor *on_value,
                               ICLTensor *output, int depth, int axis)
{
  _is_off_value_memset = true;
  ARM_COMPUTE_ERROR_ON_NULLPTR(indices, on_value, output);
  configure_common(indices, on_value, output, depth, axis);
}
void CLOneHotKernel::configure_common(const ICLTensor *indices, const ICLTensor *on_value,
                                      ICLTensor *output, int depth, int axis)
{
  ARM_COMPUTE_ERROR_THROW_ON(
      validate_arguments(indices->info(), on_value->info(), output->info(), depth, axis));
  // Configure kernel window
  auto win_config =
      validate_and_configure_window(indices->info(), on_value->info(), output->info(), depth, axis);
  ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
  if (_is_off_value_memset)
  {
    // Replace window with calculated by infices info
    win_config.second = calculate_max_window(*indices->info(), Steps());
  }
  _indices = indices;
  _on_value = on_value;
  _output = output;
  const auto actual_axis = wrap_around(axis, static_cast<int>(output->info()->num_dimensions()));
  // Set build options
  CLBuildOptions build_opts;
  build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(
                                             data_size_from_type(on_value->info()->data_type())));
  build_opts.add_option("-DAXIS=" + support::cpp11::to_string(actual_axis));
  build_opts.add_option("-DOUTPUT_DIM_Z=" +
                        support::cpp11::to_string(output->info()->dimension(2)));
  // Create kernel
  const std::string kernel_name = _is_off_value_memset ? "one_hot_only_on_value" : "one_hot";
  _kernel = static_cast<cl::Kernel>(
      CLKernelLibraryEx::get().create_kernel(kernel_name, build_opts.options()));
  ICLKernel::configure_internal(win_config.second);
}
Status CLOneHotKernel::validate(const ITensorInfo *indices, const ITensorInfo *on_value,
                                const ITensorInfo *off_value, const ITensorInfo *output, int depth,
                                int axis)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(off_value);
  ARM_COMPUTE_RETURN_ERROR_ON(off_value->tensor_shape().total_size() != 1);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(on_value, off_value);
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(indices, on_value, output, depth, axis));
  ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(indices->clone().get(),
                                                            on_value->clone().get(),
                                                            output->clone().get(), depth, axis)
                                  .first);
  return Status{};
}
Status CLOneHotKernel::validate(const ITensorInfo *indices, const ITensorInfo *on_value,
                                const ITensorInfo *output, int depth, int axis)
{
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(indices, on_value, output, depth, axis));
  ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(indices->clone().get(),
                                                            on_value->clone().get(),
                                                            output->clone().get(), depth, axis)
                                  .first);
  return Status{};
}
void CLOneHotKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
  Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
  unsigned int idx = 0;
  add_3D_tensor_argument(idx, _indices, window_collapsed);
  add_1D_tensor_argument(idx, _on_value, window_collapsed);
  if (!_is_off_value_memset)
  {
    add_1D_tensor_argument(idx, _off_value, window_collapsed);
  }
  add_4D_tensor_argument(idx, _output, window_collapsed);
  enqueue(queue, *this, window_collapsed, lws_hint());
}

} // namespace arm_compute
