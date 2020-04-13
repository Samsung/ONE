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

#include "arm_compute/core/CL/kernels/CLSpaceToBatchNDKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"

using namespace arm_compute;

namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 16;

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *block_size,
                          const ITensorInfo *padding_size, const ITensorInfo *output)
{
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::QASYMM8,
                                                       DataType::S16, DataType::F16, DataType::S32,
                                                       DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(block_size, 1, DataType::S32);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(padding_size, 1, DataType::S32);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::QASYMM8,
                                                       DataType::S16, DataType::F16, DataType::S32,
                                                       DataType::F32);

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() != output->num_dimensions(),
                                  "The number of dimensions of input should be equal to output");

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_layout() != output->data_layout(),
                                  "The input and output layouts are different!");

  // TODO Support other cases
  if (input->num_dimensions() == 4 && input->data_layout() == DataLayout::NCHW)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->dimension(2) != output->dimension(2),
                                    "Input Depth should be equal to Output Depth");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(block_size->dimension(0) != 2 ||
                                        padding_size->dimension(1) != 2,
                                    "Only 2-dimensional spatial block's size was wrong");
  }
  else if (input->num_dimensions() == 4 && input->data_layout() == DataLayout::NHWC)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->dimension(0) != output->dimension(0),
                                    "Input Depth should be equal to Output Depth");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(block_size->dimension(0) != 2 ||
                                        padding_size->dimension(1) != 2,
                                    "Only 2-dimensional spatial block's size was wrong");
  }
  else
  {
    ARM_COMPUTE_RETURN_ERROR_MSG("CLSpaceToBatchNDKernel supports only 4-dimensional input");
  }

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() < 2 && input->num_dimensions() > 4,
                                  "CLSpaceToBatchNDKernel supports dimensions up to 4");

  if (input->data_type() == DataType::QASYMM8)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->quantization_info() != output->quantization_info(),
                                    "The input and output quantization info are different!");
  }

  return Status{};
}

} // namespace

CLSpaceToBatchNDKernel::CLSpaceToBatchNDKernel()
{
  // DO NOTHING
}

void CLSpaceToBatchNDKernel::configure(const ICLTensor *input, const ICLTensor *block_size,
                                       const ICLTensor *padding_size, ICLTensor *output)
{

  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_ERROR_THROW_ON(
      validate_arguments(input->info(), block_size->info(), padding_size->info(), output->info()));

  _input = input;
  _block_size = block_size;
  _padding_size = padding_size;
  _output = output;

  // Set kernel build options
  // TODO Support other cases
  std::string kernel_name = "space_to_batch_4d";
  std::set<std::string> build_opts;
  Window win;

  if (input->info()->data_layout() == DataLayout::NCHW)
  {
    kernel_name += "_nchw";
    build_opts.emplace("-DDEPTH_OUT=" + support::cpp11::to_string(output->info()->dimension(2)));
    build_opts.emplace("-DHEIGHT_IN=" + support::cpp11::to_string(input->info()->dimension(1)));
    build_opts.emplace("-DWIDTH_IN=" + support::cpp11::to_string(input->info()->dimension(0)));

    win = calculate_max_window(*output->info(), Steps());

    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));
  }
  else if (input->info()->data_layout() == DataLayout::NHWC)
  {
    kernel_name += "_nhwc";
    build_opts.emplace("-DHEIGHT_OUT=" + support::cpp11::to_string(output->info()->dimension(2)));
    build_opts.emplace("-DHEIGHT_IN=" + support::cpp11::to_string(input->info()->dimension(2)));
    build_opts.emplace("-DWIDTH_IN=" + support::cpp11::to_string(input->info()->dimension(1)));
    build_opts.emplace("-DVEC_SIZE=" +
                       support::cpp11::to_string(num_elems_processed_per_iteration));

    win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);
    input_access.set_valid_region(win, output->info()->valid_region());

    if (window_changed)
    {
      ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!");
    }
  }
  else
  {
    ARM_COMPUTE_ERROR("Unsupported layout");
  }

  build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
  build_opts.emplace("-DBATCH_IN=" + support::cpp11::to_string(input->info()->dimension(3)));
  if (input->info()->data_type() == DataType::QASYMM8)
  {
    build_opts.emplace("-DZERO_VALUE=" + support::cpp11::to_string(
                                             input->info()->quantization_info().uniform().offset));
  }
  else
  {
    build_opts.emplace("-DZERO_VALUE=" + support::cpp11::to_string(0));
  }

  // Create kernel
  _kernel =
      static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel(kernel_name, build_opts));

  // Configure kernel window
  ICLKernel::configure_internal(win);
}

void CLSpaceToBatchNDKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

#if defined(ARM_COMPUTE_DEBUG_ENABLED)
  const_cast<ICLTensor *>(_block_size)->map(queue);
  const_cast<ICLTensor *>(_padding_size)->map(queue);

  const size_t num_dimensions = _input->info()->num_dimensions();
  const size_t num_spacial_dimensions = _block_size->info()->dimension(0);
  uint32_t batch_size = _input->info()->dimension(num_dimensions - 1);
  for (size_t i = 0; i < num_spacial_dimensions; ++i)
  {
    const int32_t block_size = *reinterpret_cast<int32_t *>(_block_size->ptr_to_element({i}));
    const int32_t padding_size_pre =
        *reinterpret_cast<int32_t *>(_padding_size->ptr_to_element({0, i}));
    const int32_t padding_size_post =
        *reinterpret_cast<int32_t *>(_padding_size->ptr_to_element({1, i}));

    ARM_COMPUTE_ERROR_ON_MSG(block_size < 1, "Block size should be greater than or equal to 1");
    ARM_COMPUTE_ERROR_ON_MSG(padding_size_pre < 0 && padding_size_post < 0,
                             "Padding size should be greater than or equal to 0");

    if (num_dimensions == 4 && _input->info()->data_layout() == DataLayout::NCHW)
    {
      ARM_COMPUTE_ERROR_ON_MSG(
          _output->info()->dimension(i) !=
              (_input->info()->dimension(i) + padding_size_pre + padding_size_post) / block_size,
          "Dimension value of spatial block does not match output's dimension value");
    }
    else
    {
      ARM_COMPUTE_ERROR_ON_MSG(
          _output->info()->dimension(num_dimensions - num_spacial_dimensions - 1 + i) !=
              (_input->info()->dimension(num_dimensions - num_spacial_dimensions - 1 + i) +
               padding_size_pre + padding_size_post) /
                  block_size,
          "Dimension value of spatial block does not match output's dimension value");
    }

    batch_size *= block_size;
  }
  ARM_COMPUTE_ERROR_ON_MSG(
      _output->info()->dimension(num_dimensions - 1) != batch_size,
      "Output batch size should be equal to input batch size * (multiplication of all block size)");

  const_cast<ICLTensor *>(_block_size)->unmap(queue);
  const_cast<ICLTensor *>(_padding_size)->unmap(queue);
#endif // defined(ARM_COMPUTE_DEBUG_ENABLED)

  Window slice_out = window.first_slice_window_4D().collapse(ICLKernel::window(), 2, 4);

  // Setup output slice
  Window slice_in(slice_out);
  slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
  slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
  slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));
  slice_in.set(3, Window::Dimension(0, 0, 0));

  // Set block size window
  Window win_block = calculate_max_window(*_block_size->info(), Steps());

  // Set padding size window
  Window win_padding = calculate_max_window(*_padding_size->info(), Steps());

  do
  {
    unsigned int idx = 0;
    add_4D_tensor_argument(idx, _input, slice_in);
    add_4D_tensor_argument(idx, _output, slice_out);
    add_1D_tensor_argument(idx, _block_size, win_block);
    add_2D_tensor_argument(idx, _padding_size, win_padding);
    enqueue(queue, *this, slice_out);
  } while (window.slide_window_slice_4D(slice_out) && window.slide_window_slice_4D(slice_in));
}
