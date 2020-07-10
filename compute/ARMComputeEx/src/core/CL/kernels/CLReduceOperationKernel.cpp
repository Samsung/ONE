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

#include "arm_compute/core/CL/kernels/CLReduceOperationKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "support/StringSupport.h"

using namespace arm_compute;
namespace
{
// NOTE This is necessary because it is not guaranteed that the axis positions of input and output
// are the same.
const TensorShape inferOutputShape(const TensorShape &input_shape, const uint32_t axis)
{
  TensorShape out_shape{input_shape};

  out_shape.set(axis, 1);

  return out_shape;
}
} // namespace

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const uint32_t axis,
                          ReduceOperation op)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

  if (output->total_size() != 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
  }

  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16,
                                                       DataType::F32, DataType::S32);
  if (op == ReduceOperation::SUM)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QASYMM8,
                                    "Not support QASYMM8, yet");
  }
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->tensor_shape().total_size() == 0,
                                  "Inputs are not broadcast compatible");

  const auto num_dimensions = input->tensor_shape().num_dimensions();
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= num_dimensions, "axis must be less than (input's rank).");

  const TensorShape output_shape = inferOutputShape(input->tensor_shape(), axis);
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(output_shape.total_size() != output->tensor_shape().total_size(),
                                  "output shape's size does not match axis");

  return Status{};
}
} // namespace

CLReduceOperationKernel::CLReduceOperationKernel() : _input(nullptr), _output(nullptr), _axis() {}

void CLReduceOperationKernel::configure(const ICLTensor *input, ICLTensor *output,
                                        const uint32_t axis, ReduceOperation op)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis, op));

  _input = input;
  _output = output;
  _axis = axis;

  std::unique_ptr<ITensorInfo> output_info = output->info()->clone();
  output_info->set_tensor_shape(inferOutputShape(input->info()->tensor_shape(), axis));

  // Construct kernel name
  std::string kernel_name;
  int op_code = 0;
  if (op == ReduceOperation::MAX)
  {
    kernel_name = "reduce_min_max";
    op_code = 1;
  }
  else if (op == ReduceOperation::MIN)
  {
    kernel_name = "reduce_min_max";
    op_code = 2;
  }
  else if (op == ReduceOperation::SUM)
  {
    kernel_name = "reduce_sum_mean";
    op_code = 3;
  }
  else if (op == ReduceOperation::MEAN)
  {
    kernel_name = "reduce_sum_mean";
    op_code = 4;
  }
  else
    throw std::runtime_error("Operation not supported, yet");

  // Set kernel build options
  std::set<std::string> build_opts;
  build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(output_info->data_type()));
  build_opts.emplace("-DDEPTH_OUT=" + support::cpp11::to_string(output_info->dimension(2)));
  build_opts.emplace("-DOP_CODE=" + support::cpp11::to_string(op_code));

  // Create kernel
  _kernel =
      static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel(kernel_name, build_opts));

  // Configure  kernel window
  Window win = calculate_max_window(*output_info, Steps());

  Coordinates coord;
  coord.set_num_dimensions(output_info->num_dimensions());
  output->info()->set_valid_region(ValidRegion(coord, output_info->tensor_shape()));

  ICLKernel::configure_internal(win);
}

Status CLReduceOperationKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                         const uint32_t axis, ReduceOperation op)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis, op));

  return Status{};
}

void CLReduceOperationKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

  const TensorShape &shape_in = _input->info()->tensor_shape();

  unsigned int idx = 2 * num_arguments_per_4D_tensor(); // Skip the input and output parameters

  _kernel.setArg<cl_int>(idx++, _axis);
  _kernel.setArg<cl_int>(idx++, shape_in[_axis]);

  // Support dimensions up to 4
  Window slice_out = window.collapse(ICLKernel::window(), 2, 4);

  // Setup input slice
  Window slice_in(slice_out);
  slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
  slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
  slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));
  slice_in.set(3, Window::Dimension(0, 0, 0));

  // Copy output's shape in order to use for recovering at end of this method
  // TODO Remove changing and recovering output's shape if it is guaranteed that the axis positions
  // of input and output are the same
  const TensorShape shape_out = _output->info()->tensor_shape();
  _output->info()->set_tensor_shape(inferOutputShape(shape_in, _axis));

  idx = 0;
  add_4D_tensor_argument(idx, _input, slice_in);
  add_4D_tensor_argument(idx, _output, slice_out);
  enqueue(queue, *this, slice_out, lws_hint());

  // Recover output's shape of output tensor
  _output->info()->set_tensor_shape(shape_out);
}
