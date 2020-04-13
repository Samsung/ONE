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

#include "arm_compute/core/CL/kernels/CLDepthToSpaceKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"

using namespace arm_compute;

namespace
{
// TODO Use this validation function
#if 0
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output,
                          const int32_t block_size)
{
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::QASYMM8,
                                                DataType::S16, DataType::S32, DataType::F16,
                                                DataType::F32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::QASYMM8,
                                                DataType::S16, DataType::S32, DataType::F16,
                                                DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(block_size < 1,
                                  "Block size should be greater than or equal to 1.");

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(0) != input->dimension(0) * block_size,
                                  "Output width should be equal to (Input width * block size)");

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(1) != input->dimension(1) * block_size,
                                  "Output height should be equal to (Input height * block size)");

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->dimension(2) % (block_size * block_size) != 0,
                                  "Input depth should be divisible by (block size * block size)");

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(
      output->dimension(2) != input->dimension(2) / (block_size * block_size),
      "Output depth should be equal to (Input depth / (block size * block size))");

  return Status{};
}
#endif
} // namespace

CLDepthToSpaceKernel::CLDepthToSpaceKernel() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void CLDepthToSpaceKernel::configure(const ICLTensor *input, ICLTensor *output,
                                     const int32_t block_size)
{
  // TODO Add validation of data_layout
  _input = input;
  _output = output;

  // Set kernel build options
  auto layout_out = output->info()->data_layout();
  std::set<std::string> build_opts;
  build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
  build_opts.emplace("-DBLOCK_SIZE=" + support::cpp11::to_string(block_size));
  auto index_depth = get_data_layout_dimension_index(layout_out, DataLayoutDimension::CHANNEL);
  auto depth = output->info()->dimension(index_depth);
  build_opts.emplace("-DDEPTH_OUT=" + support::cpp11::to_string(depth));
  build_opts.emplace("-DZ_OUT=" + support::cpp11::to_string(output->info()->tensor_shape().z()));

  // Create kernel
  _kernel = static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel(
      "depth_to_space_" + lower_string(string_from_data_layout(layout_out)), build_opts));

  // Configure  kernel window
  Window win = calculate_max_window(*output->info(), Steps());

  Coordinates coord;
  coord.set_num_dimensions(output->info()->num_dimensions());
  output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

  ICLKernel::configure_internal(win);
}

void CLDepthToSpaceKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

  Window slice_out = window.first_slice_window_4D().collapse(ICLKernel::window(), 2, 4);

  // Setup input slice
  Window slice_in(slice_out);
  slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
  slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
  slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));
  slice_in.set(3, Window::Dimension(0, 0, 0));

  do
  {
    unsigned int idx = 0;
    add_4D_tensor_argument(idx, _input, slice_in);
    add_4D_tensor_argument(idx, _output, slice_out);
    enqueue(queue, *this, slice_out);
  } while (window.slide_window_slice_4D(slice_in) && window.slide_window_slice_4D(slice_out));
}
