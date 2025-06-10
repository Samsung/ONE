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
 * Copyright (c) 2019 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEOneHotKernel.h"
#include "src/core/CPP/Validate.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculatorEx.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/helpers/AutoConfiguration.h"

namespace arm_compute
{
namespace
{
/** Validate the depth
 *
 * Validate that depth are not negative
 *
 * @param[in] depth Depth tensor.
 * @param[in] output Output tensor.
 * @param[in] axis Axis of depth.
 */
template <typename U> void validate_depth(const ITensor *depth, const ITensor *output, int axis)
{
  ARM_COMPUTE_ERROR_ON(*(reinterpret_cast<U *>(depth->buffer())) < 0);
  ARM_COMPUTE_ERROR_ON(static_cast<U>(output->info()->tensor_shape()[axis]) !=
                       *(reinterpret_cast<U *>(depth->buffer())));
}

Status validate_arguments(const ITensorInfo *indices, const ITensorInfo *depth,
                          const ITensorInfo *on_value, const ITensorInfo *off_value,
                          const ITensorInfo *output, int axis)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(indices, depth, on_value, off_value, output);
  const int actual_axis = wrap_around(axis, static_cast<int>(output->num_dimensions()));
  ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() > 4);
  ARM_COMPUTE_RETURN_ERROR_ON(on_value->tensor_shape().total_size() != 1);
  ARM_COMPUTE_RETURN_ERROR_ON(0 > actual_axis ||
                              actual_axis >= static_cast<int>(output->num_dimensions()));
  ARM_COMPUTE_RETURN_ERROR_ON(on_value->data_type() == DataType::UNKNOWN);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(on_value, 1, DataType::U8, DataType::S8,
                                                       DataType::U16, DataType::S16, DataType::F16,
                                                       DataType::U32, DataType::S32, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32, DataType::S32);

  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(on_value, off_value);
  if (output->total_size() != 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(on_value, output);
  }

  return Status{};
}

template <typename U, typename Enable = void> bool isOnValue(U) { return true; }

template <typename U, std::enable_if_t<std::is_integral<U>::value, int> = 0>
bool isOnValue(U index, U depth)
{
  return index >= 0 && index < depth;
}
} // namespace

NEOneHotKernel::NEOneHotKernel()
  : _indices{nullptr}, _depth{nullptr}, _on_value{nullptr}, _off_value{nullptr}, _axis{-1},
    _output{nullptr}, _func{}
{
}

template <typename U>
void NEOneHotKernel::onehot_0_axis(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  // Validate that the depth are not negative
  validate_depth<U>(_depth, _output, _axis);
  Window output_window{window};
  output_window.set(Window::DimX, Window::Dimension(0, 1, 1));
  Iterator output_it(_output, output_window);
  const U off_value = *reinterpret_cast<U *>(_off_value->buffer());
  execute_window_loop(
    output_window,
    [&](const Coordinates &id) {
      std::fill_n(output_it.ptr(), _output->info()->dimension(0) * _output->info()->element_size(),
                  off_value);
      Coordinates indices_id(id);
      indices_id.remove(0);
      const U new_index = *(reinterpret_cast<U *>(_indices->ptr_to_element(indices_id)));
      if (isOnValue(new_index, *(reinterpret_cast<U *>(_depth->buffer()))))
      {
        Coordinates onehot_id(id);
        onehot_id.set(0, new_index);
        std::copy_n(_on_value->buffer(), _output->info()->element_size(),
                    _output->ptr_to_element(onehot_id));
      }
    },
    output_it);
}

template <typename U>
inline void NEOneHotKernel::onehot_n_axis(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  // Validate that the indices are not negative
  validate_depth<U>(_depth, _output, _axis);
  Iterator output_it(_output, window);
  execute_window_loop(
    window,
    [&](const Coordinates &id) {
      Coordinates indices_id(id);
      indices_id.remove(_axis);
      const U new_index = *(reinterpret_cast<U *>(_indices->ptr_to_element(indices_id)));
      if (isOnValue(new_index, *(reinterpret_cast<U *>(_depth->buffer()))))
      {
        Coordinates onehot_id(id);
        onehot_id.set(_axis, new_index);
        std::copy_n(static_cast<U>(id[_axis]) == new_index ? _on_value->buffer()
                                                           : _off_value->buffer(),
                    _output->info()->element_size(), output_it.ptr());
      }
    },
    output_it);
}

void NEOneHotKernel::configure(const ITensor *indices, const ITensor *depth,
                               const ITensor *on_value, const ITensor *off_value, ITensor *output,
                               int axis)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(indices, depth, on_value, off_value, output);
  ARM_COMPUTE_ERROR_ON(output->info()->total_size() == 0);
  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(indices->info(), depth->info(), on_value->info(),
                                                off_value->info(), output->info(), axis));
  _indices = indices;
  _depth = depth;
  _on_value = on_value;
  _off_value = off_value;
  _output = output;
  _axis = wrap_around(axis, static_cast<int>(output->info()->num_dimensions()));
  if (0 == _axis)
  {
    switch (_indices->info()->data_type())
    {
      case DataType::U32:
        _func = &NEOneHotKernel::onehot_0_axis<uint32_t>;
        break;
      case DataType::S32:
        _func = &NEOneHotKernel::onehot_0_axis<int32_t>;
        break;
      default:
        ARM_COMPUTE_ERROR("Not supported");
        break;
    }
  }
  else
  {
    switch (_indices->info()->data_type())
    {
      case DataType::U32:
        _func = &NEOneHotKernel::onehot_n_axis<uint32_t>;
        break;
      case DataType::S32:
        _func = &NEOneHotKernel::onehot_n_axis<int32_t>;
        break;
      default:
        ARM_COMPUTE_ERROR("Not supported");
        break;
    }
  }
  // Create window
  Window win = calculate_max_window(*output->info(), Steps());
  output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));
  INEKernel::configure(win);
}

Status NEOneHotKernel::validate(const ITensorInfo *indices, const ITensorInfo *depth,
                                const ITensorInfo *on_value, const ITensorInfo *off_value,
                                const ITensorInfo *output, int axis)
{
  ARM_COMPUTE_RETURN_ON_ERROR(
    validate_arguments(indices, depth, on_value, off_value, output, axis));
  return Status{};
}

void NEOneHotKernel::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON(_func == nullptr);
  (this->*_func)(window, info);
}
} // namespace arm_compute
