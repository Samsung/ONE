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

#include "arm_compute/core/CL/kernels/CLBinaryLogicalOpKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"

using namespace arm_compute;

namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 16;

Status validate_parameters(const ITensorInfo *input1, const ITensorInfo *input2,
                           const ITensorInfo *output)
{
  const TensorShape &out_shape =
      TensorShape::broadcast_shape(input1->tensor_shape(), input2->tensor_shape());

  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8, DataType::QASYMM8);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::U8, DataType::QASYMM8);

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0,
                                  "Inputs are not broadcast compatible");
  // Validate in case of configured output
  if (output->total_size() > 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8,
                                                         DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        detail::have_different_dimensions(out_shape, output->tensor_shape(), 0),
        "Wrong shape for output");
  }
  return Status{};
}
} // namespace

CLBinaryLogicalOpKernel::CLBinaryLogicalOpKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void CLBinaryLogicalOpKernel::configure(const ICLTensor *input1, const ICLTensor *input2,
                                        ICLTensor *output, BinaryLogicalOperation op)
{
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input1, input2);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input1, output);
  ARM_COMPUTE_ERROR_THROW_ON(validate_parameters(input1->info(), input2->info(), output->info()));

  _input1 = input1;
  _input2 = input2;
  _output = output;

  // Create kernel
  std::string kernel_name = "binary_logical_op";
  std::set<std::string> build_opts;
  build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input1->info()->data_type())));

  int op_code = 0;
  switch (op)
  {
    case BinaryLogicalOperation::AND:
      op_code = 1;
      break;
    case BinaryLogicalOperation::OR:
      op_code = 2;
      break;
    default:
      throw std::runtime_error("Operation not supported, yet");
  }

  build_opts.emplace(("-DOP_CODE=" + support::cpp11::to_string(op_code)));
  build_opts.emplace(
      ("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration)));

  _kernel =
      static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel(kernel_name, build_opts));

  const std::pair<TensorShape, ValidRegion> broadcast_pair =
      ITensorInfo::broadcast_shape_and_valid_region(*input1->info(), *input2->info());

  const ValidRegion &valid_region = broadcast_pair.second;

  Window win = calculate_max_window(valid_region, Steps(num_elems_processed_per_iteration));
  Window win_input1 = win.broadcast_if_dimension_le_one(*input1->info());
  Window win_input2 = win.broadcast_if_dimension_le_one(*input2->info());

  AccessWindowHorizontal input1_access(input1->info(), 0, num_elems_processed_per_iteration);
  AccessWindowHorizontal input2_access(input2->info(), 0, num_elems_processed_per_iteration);
  AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

  update_window_and_padding(win_input1, input1_access) ||
      update_window_and_padding(win_input2, input2_access) ||
      update_window_and_padding(win, output_access);

  output_access.set_valid_region(win, valid_region);

  ICLKernel::configure_internal(win);
}

void CLBinaryLogicalOpKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

  const TensorShape &in_shape1 = _input1->info()->tensor_shape();
  const TensorShape &in_shape2 = _input2->info()->tensor_shape();
  const TensorShape &out_shape = _output->info()->tensor_shape();

  bool can_collapse = true;
  if (std::min(in_shape1.total_size(), in_shape2.total_size()) > 1)
  {
    can_collapse =
        (std::min(in_shape1.num_dimensions(), in_shape2.num_dimensions()) > Window::DimZ);
    for (size_t d = Window::DimZ; can_collapse && (d < out_shape.num_dimensions()); d++)
    {
      can_collapse = (in_shape1[d] == in_shape2[d]);
    }
  }

  bool has_collapsed = false;
  Window collapsed =
      can_collapse ? window.collapse_if_possible(ICLKernel::window(), Window::DimZ, &has_collapsed)
                   : window;

  const TensorShape &in_shape1_collapsed =
      has_collapsed ? in_shape1.collapsed_from(Window::DimZ) : in_shape1;
  const TensorShape &in_shape2_collapsed =
      has_collapsed ? in_shape2.collapsed_from(Window::DimZ) : in_shape2;

  Window slice = collapsed.first_slice_window_3D();
  Window slice_input1 = slice.broadcast_if_dimension_le_one(in_shape1_collapsed);
  Window slice_input2 = slice.broadcast_if_dimension_le_one(in_shape2_collapsed);

  do
  {
    unsigned int idx = 0;
    add_3D_tensor_argument(idx, _input1, slice_input1);
    add_3D_tensor_argument(idx, _input2, slice_input2);
    add_3D_tensor_argument(idx, _output, slice);

    enqueue(queue, *this, slice);

    collapsed.slide_window_slice_3D(slice_input1);
    collapsed.slide_window_slice_3D(slice_input2);
  } while (collapsed.slide_window_slice_3D(slice));
}

BorderSize CLBinaryLogicalOpKernel::border_size() const
{
  const unsigned int replicateSize =
      _output->info()->dimension(0) -
      std::min(_input1->info()->dimension(0), _input2->info()->dimension(0));
  const unsigned int border =
      std::min<unsigned int>(num_elems_processed_per_iteration - 1U, replicateSize);
  return BorderSize(0, border, 0, 0);
}
