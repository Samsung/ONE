/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * Copyright (c) 2018-2019 ARM Limited.
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

#include "arm_compute/core/NEON/kernels/NEHashtableLookupKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <unordered_map>

using namespace arm_compute;

namespace
{
constexpr size_t NOT_HIT = 0xFFFFFFFF;
} // namespace

NEHashtableLookupKernel::NEHashtableLookupKernel()
    : _lookups(nullptr), _keys(nullptr), _input(nullptr), _output(nullptr), _hits{nullptr}
{
}

void NEHashtableLookupKernel::configure(const ITensor *lookups, const ITensor *keys,
                                        const ITensor *input, ITensor *output, ITensor *hits)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(lookups, keys, input, output, hits);
  ARM_COMPUTE_ERROR_THROW_ON(
      validate(lookups->info(), keys->info(), input->info(), output->info(), hits->info()));

  _lookups = lookups;
  _keys = keys;
  _input = input;
  _output = output;
  _hits = hits;

  // Auto initialize output if not initialized
  auto out_shape{input->info()->tensor_shape()};
  out_shape.set(out_shape.num_dimensions() - 1, lookups->info()->num_dimensions(), false);
  auto_init_if_empty(*output->info(), out_shape, 1, input->info()->data_type(),
                     input->info()->quantization_info());

  // Auto initialize hits if not initialized
  auto_init_if_empty(*hits->info(), lookups->info()->tensor_shape(), 1, DataType::U8);

  INEKernel::configure(calculate_max_window(*output->info()));
}

Status NEHashtableLookupKernel::validate(const ITensorInfo *lookups, const ITensorInfo *keys,
                                         const ITensorInfo *input, const ITensorInfo *output,
                                         const ITensorInfo *hits)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(lookups, keys, input, output, hits);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(
      input, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::U16, DataType::S16,
      DataType::U32, DataType::S32, DataType::F16, DataType::F32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lookups, 1, DataType::S32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(keys, 1, DataType::S32);

  ARM_COMPUTE_ERROR_ON(input->num_dimensions() < 2 && input->num_dimensions() > 4);
  ARM_COMPUTE_ERROR_ON(lookups->num_dimensions() > 1);
  ARM_COMPUTE_ERROR_ON(keys->num_dimensions() > 1);
  ARM_COMPUTE_ERROR_ON(keys->dimension(0) != input->dimension(input->num_dimensions() - 1));

  // Validate in case of configured output
  if (output->total_size() > 0)
  {
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON(input->num_dimensions() != output->num_dimensions());
    ARM_COMPUTE_ERROR_ON(output->dimension(output->num_dimensions() - 1) != lookups->dimension(0));
    for (size_t i = 0; i < output->num_dimensions() - 1; ++i)
    {
      ARM_COMPUTE_ERROR_ON(input->dimension(i) != output->dimension(i));
    }
  }

  // Validate in case of configured hits
  if (hits->total_size() > 0)
  {
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(hits, 1, DataType::U8, DataType::QASYMM8);
    ARM_COMPUTE_ERROR_ON(hits->dimension(0) != output->dimension(output->num_dimensions() - 1));
    ARM_COMPUTE_ERROR_ON(hits->dimension(0) != lookups->dimension(0));
    ARM_COMPUTE_ERROR_ON(hits->num_dimensions() > 1);
  }

  return Status{};
}

void NEHashtableLookupKernel::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

  const size_t lookup_dim = _output->info()->num_dimensions() - 1;
  const int const_0 = _output->info()->data_type() == DataType::QASYMM8
                          ? _output->info()->quantization_info().uniform().offset
                          : 0;

  std::unordered_map<int32_t, size_t> key_index_map;
  for (size_t n = 0; n < _keys->info()->dimension(0); ++n)
  {
    const int32_t key = *reinterpret_cast<int32_t *>(_keys->ptr_to_element({n}));
    key_index_map[key] = n;
  }
  std::vector<size_t> lookup_indices;
  for (size_t k = 0; k < _lookups->info()->dimension(0); ++k)
  {
    const int32_t key = *reinterpret_cast<int32_t *>(_lookups->ptr_to_element({k}));
    const auto it = key_index_map.find(key);
    if (it == key_index_map.end())
    {
      lookup_indices.emplace_back(NOT_HIT);
      *_hits->ptr_to_element({k}) = 0;
    }
    else
    {
#if defined(ARM_COMPUTE_DEBUG_ENABLED)
      if (it->second >= _keys->info()->dimension(0))
        ARM_COMPUTE_ERROR("HashTable Lookup: Index out of bounds.");
#endif // defined(ARM_COMPUTE_DEBUG_ENABLED)
      lookup_indices.emplace_back(it->second);
      *_hits->ptr_to_element({k}) = 1;
    }
  }

  Window output_window{window};
  output_window.set(Window::DimX,
                    Window::Dimension(output_window.x().start(), output_window.x().end(),
                                      _input->info()->dimension(0)));

  Window out_slice = output_window.first_slice_window_4D();
  do
  {
    Iterator output_it(_output, out_slice);

    execute_window_loop(
        out_slice,
        [&](const Coordinates &id) {
          const auto lookup = lookup_indices.at(id[lookup_dim]);
          if (lookup == NOT_HIT)
          {
            memset(output_it.ptr(), const_0,
                   _output->info()->dimension(0) * _output->info()->element_size());
          }
          else
          {
            Coordinates input_id{id};
            input_id.set(lookup_dim, lookup);
            memcpy(output_it.ptr(), _input->ptr_to_element(input_id),
                   _output->info()->dimension(0) * _output->info()->element_size());
          }
        },
        output_it);

  } while (window.slide_window_slice_4D(out_slice));
}
