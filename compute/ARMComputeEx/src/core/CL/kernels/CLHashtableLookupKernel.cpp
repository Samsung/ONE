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

#include "arm_compute/core/CL/kernels/CLHashtableLookupKernel.h"

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

CLHashtableLookupKernel::CLHashtableLookupKernel()
{
  // DO NOTHING
}

Status CLHashtableLookupKernel::validate(const ITensorInfo *lookups, const ITensorInfo *keys,
                                         const ITensorInfo *input, const ITensorInfo *output,
                                         const ITensorInfo *hits)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(lookups, keys, input, output, hits);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(
    input, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::U16, DataType::S16,
    DataType::U32, DataType::S32, DataType::F16, DataType::F32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lookups, 1, DataType::S32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(keys, 1, DataType::S32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(hits, 1, DataType::U8, DataType::QASYMM8);
  ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->tensor_shape().total_size() == 0,
                                  "Output's shape was not set");

  ARM_COMPUTE_ERROR_ON(lookups->dimension(0) != hits->dimension(0) ||
                       output->dimension(output->num_dimensions() - 1) != lookups->dimension(0));
  ARM_COMPUTE_ERROR_ON(input->num_dimensions() < 2 && input->num_dimensions() > 4);
  ARM_COMPUTE_ERROR_ON(lookups->num_dimensions() > 1);
  ARM_COMPUTE_ERROR_ON(keys->num_dimensions() > 1);
  ARM_COMPUTE_ERROR_ON(hits->num_dimensions() > 1);

  return Status{};
}

void CLHashtableLookupKernel::configure(const ICLTensor *lookups, const ICLTensor *keys,
                                        const ICLTensor *input, ICLTensor *output, ICLTensor *hits)
{
  ARM_COMPUTE_ERROR_THROW_ON(
    validate(lookups->info(), keys->info(), input->info(), output->info(), hits->info()));

  _lookups = lookups;
  _keys = keys;
  _input = input;
  _output = output;
  _hits = hits;

  // Make _lookup_indices tensor
  _lookup_indices = support::cpp14::make_unique<CLTensor>();
  _lookup_indices->allocator()->init(
    TensorInfo(lookups->info()->tensor_shape(), lookups->info()->num_channels(), DataType::S32));
  _lookup_indices->allocator()->allocate();

  // Set kernel build options
  std::stringstream kernel_name;
  std::set<std::string> build_opts;
  kernel_name << "hashtable_lookup";

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

void CLHashtableLookupKernel::run(const Window &window, cl::CommandQueue &queue)
{
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

  const_cast<ICLTensor *>(_lookups)->map(queue);
  const_cast<ICLTensor *>(_keys)->map(queue);
  _hits->map(queue);
  _lookup_indices->map(queue);

  // Set values of hits
  const int32_t *lookups_buf =
    reinterpret_cast<int32_t *>(const_cast<ICLTensor *>(_lookups)->buffer());
  const int32_t *keys_buf = reinterpret_cast<int32_t *>(const_cast<ICLTensor *>(_keys)->buffer());
  uint8_t *hits_buf = reinterpret_cast<uint8_t *>(_hits->buffer());
  int32_t *lookup_indices_buf = reinterpret_cast<int32_t *>(_lookup_indices->buffer());

  std::map<int32_t, size_t> key_map;
  const size_t keys_num = _keys->info()->dimension(0);
  for (size_t key_index = 0; key_index < keys_num; key_index++)
  {
    key_map[keys_buf[key_index]] = key_index;
  }

  const size_t lookups_num = _lookups->info()->dimension(0);
  for (size_t i = 0; i < lookups_num; ++i)
  {
    const auto lookup_value = lookups_buf[i];
    const auto it = key_map.find(lookup_value);
    if (it != key_map.end())
    {
#if defined(ARM_COMPUTE_DEBUG_ENABLED)
      if (it->second >= lookups_num)
        ARM_COMPUTE_ERROR("HashTable Lookup: index out of bounds.");
#endif // defined(ARM_COMPUTE_DEBUG_ENABLED)
      lookup_indices_buf[i] = static_cast<int32_t>(it->second);
      hits_buf[i] = static_cast<uint8_t>(1);
    }
    else
    {
      lookup_indices_buf[i] = -1;
      hits_buf[i] = static_cast<uint8_t>(0);
    }
  }

  const_cast<ICLTensor *>(_lookups)->unmap(queue);
  const_cast<ICLTensor *>(_keys)->unmap(queue);
  _hits->unmap(queue);
  _lookup_indices->unmap(queue);

  Window win = window.collapse(ICLKernel::window(), 2, 4);

  Window win_lookup;
  win_lookup.set(Window::DimX, Window::Dimension(0, 0, 0));

  do
  {
    unsigned int idx = 0;
    add_4D_tensor_argument(idx, _input, win);
    add_4D_tensor_argument(idx, _output, win);
    add_1D_tensor_argument(idx, _lookup_indices.get(), win_lookup);

    enqueue(queue, *this, win);
  } while (window.slide_window_slice_4D(win) && window.slide_window_slice_1D(win_lookup));
}
