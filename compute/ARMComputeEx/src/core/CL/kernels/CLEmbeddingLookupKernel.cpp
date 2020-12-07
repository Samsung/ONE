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
 * Copyright (c) 2017 ARM Limited.
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

#include "arm_compute/core/CL/kernels/CLEmbeddingLookupKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "support/StringSupport.h"

using namespace arm_compute;

namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 16;

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
  Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));
  AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
  AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

  bool window_changed = update_window_and_padding(win, input_access, output_access);
  input_access.set_valid_region(win, output->valid_region());

  Status err = (window_changed)
                 ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!")
                 : Status{};
  return std::make_pair(err, win);
}
} // namespace

CLEmbeddingLookupKernel::CLEmbeddingLookupKernel()
  : _input(nullptr), _output(nullptr), _lookups(nullptr)
{
}

Status CLEmbeddingLookupKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                         const ITensorInfo *lookups)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, lookups);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(
    input, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::U16, DataType::S16,
    DataType::U32, DataType::S32, DataType::F16, DataType::F32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lookups, 1, DataType::S32);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

  ARM_COMPUTE_ERROR_ON(input->num_dimensions() < 2 && input->num_dimensions() > 4);
  ARM_COMPUTE_ERROR_ON(lookups->num_dimensions() > 1);

  return Status{};
}

void CLEmbeddingLookupKernel::configure(const ICLTensor *input, ICLTensor *output,
                                        const ICLTensor *lookups)
{
  ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), lookups->info()));

  _input = input;
  _output = output;
  _lookups = lookups;

  // Set kernel build options
  std::stringstream kernel_name;
  std::set<std::string> build_opts;
  kernel_name << "embedding_lookup";

  build_opts.emplace("-DDEPTH_OUT=" + support::cpp11::to_string(output->info()->dimension(2)));
  build_opts.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
  build_opts.emplace("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
  build_opts.emplace("-DNUM_DIMS=" + support::cpp11::to_string(_input->info()->num_dimensions()));

  // Create kernel
  _kernel =
    static_cast<cl::Kernel>(CLKernelLibraryEx::get().create_kernel(kernel_name.str(), build_opts));

  // Configure kernel window
  auto win_config = validate_and_configure_window(input->info(), output->info());
  ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
  ICLKernel::configure_internal(win_config.second);
}

void CLEmbeddingLookupKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  Window slice_in = window.first_slice_window_4D().collapse(ICLKernel::window(), 2, 4);

  Window win_lookup;
  win_lookup.set(Window::DimX, Window::Dimension(0, 0, 0));

  do
  {
    unsigned int idx = 0;
    add_4D_tensor_argument(idx, _input, slice_in);
    add_4D_tensor_argument(idx, _output, slice_in);
    add_1D_tensor_argument(idx, _lookups, win_lookup);

    enqueue(queue, *this, slice_in);
  } while (window.slide_window_slice_4D(slice_in) && window.slide_window_slice_1D(win_lookup));
}
