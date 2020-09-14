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
 * Copyright (c) 2019 ARM Limited.
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

#include "arm_compute/core/NEON/kernels/NEGatherKernelEx.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculatorEx.h"

namespace arm_compute
{
namespace
{
/** Validate the indices
 *
 * Validate that indices are not negative
 *
 * @param[in] indices Indices tensor info.
 */
template <typename U> void validate_indices(const ITensor *indices)
{
  for (size_t i = 0; i < indices->info()->tensor_shape()[0]; ++i)
  {
    ARM_COMPUTE_ERROR_ON(*(reinterpret_cast<U *>(indices->ptr_to_element(Coordinates(i)))) < 0);
  }
}

} // namespace

NEGatherKernelEx::NEGatherKernelEx()
    : _input{}, _indices{}, _axis{}, _indices_rank{}, _output{}, _func{}
{
}

template <typename U>
inline void NEGatherKernelEx::gather_0_axis(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);

  // Validate that the indices are not negative
  validate_indices<U>(_indices);

  Iterator output_it(_output, window);
  execute_window_loop(
      window,
      [&](const Coordinates &id) {
        Coordinates gather_id(id);
        gather_id.collapse(_indices_rank);

        U new_index;
        switch (_indices_rank)
        {
          case 1:
            new_index = *(reinterpret_cast<U *>(_indices->ptr_to_element(Coordinates(id[0]))));
            break;
          case 2:
            new_index =
                *(reinterpret_cast<U *>(_indices->ptr_to_element(Coordinates(id[0], id[1]))));
            break;
          case 3:
            new_index = *(
                reinterpret_cast<U *>(_indices->ptr_to_element(Coordinates(id[0], id[1], id[2]))));
            break;
          default:
            ARM_COMPUTE_ERROR("Wrong num of dimensions");
            break;
        }

        gather_id.set(0, new_index);

        std::copy_n(_input->ptr_to_element(gather_id), _output->info()->element_size(),
                    output_it.ptr());
      },
      output_it);
}

template <typename U>
void NEGatherKernelEx::gather_n_axis(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);

  // Validate that the indices are not negative
  validate_indices<U>(_indices);

  Window output_window{window};
  output_window.set(Window::DimX, Window::Dimension(0, 1, 1));

  Iterator output_it(_output, output_window);
  execute_window_loop(
      output_window,
      [&](const Coordinates &id) {
        Coordinates gather_id(id);
        gather_id.collapse(_indices_rank, _axis);

        U new_index;
        switch (_indices_rank)
        {
          case 1:
            new_index = *(reinterpret_cast<U *>(_indices->ptr_to_element(Coordinates(id[_axis]))));
            break;
          case 2:
            new_index = *(reinterpret_cast<U *>(
                _indices->ptr_to_element(Coordinates(id[_axis], id[_axis + 1]))));
            break;
          case 3:
            new_index = *(reinterpret_cast<U *>(
                _indices->ptr_to_element(Coordinates(id[_axis], id[_axis + 1], id[_axis + 2]))));
            break;
          default:
            ARM_COMPUTE_ERROR("Wrong num of dimensions");
            break;
        }

        gather_id.set(_axis, new_index);

        std::copy_n(_input->ptr_to_element(gather_id),
                    _input->info()->dimension(0) * _output->info()->element_size(),
                    output_it.ptr());
      },
      output_it);
}

void NEGatherKernelEx::configure(const ITensor *input, const ITensor *indices, ITensor *output,
                                 int axis)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, indices);
  ARM_COMPUTE_ERROR_ON(indices->info()->num_dimensions() > 3);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32, DataType::S32);
  ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(
      input, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::U16, DataType::S16,
      DataType::U32, DataType::S32, DataType::F16, DataType::F32);

  _input = input;
  _indices = indices;
  _output = output;
  _axis = axis;
  _indices_rank = indices->info()->num_dimensions();

  if (_axis < 0)
  {
    _axis += input->info()->num_dimensions();
  }
  ARM_COMPUTE_ERROR_ON(0 > _axis || _axis >= static_cast<int32_t>(input->info()->num_dimensions()));

  if (0 == _axis)
  {
    switch (_indices->info()->data_type())
    {
      case DataType::U32:
        _func = &NEGatherKernelEx::gather_0_axis<uint32_t>;
        break;
      case DataType::S32:
        _func = &NEGatherKernelEx::gather_0_axis<int32_t>;
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
        _func = &NEGatherKernelEx::gather_n_axis<uint32_t>;
        break;
      case DataType::S32:
        _func = &NEGatherKernelEx::gather_n_axis<int32_t>;
        break;
      default:
        ARM_COMPUTE_ERROR("Not supported");
        break;
    }
  }
  // Output auto initialization if not yet initialized
  TensorShape output_shape = arm_compute::misc::shape_calculator::compute_gather_shape_ex(
      input->info()->tensor_shape(), indices->info()->tensor_shape(), _axis);
  auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type());

  // Create window
  Window win = calculate_max_window(*output->info(), Steps());
  output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

  INEKernel::configure(win);
}

Status NEGatherKernelEx::validate(const ITensorInfo *input, const ITensorInfo *indices,
                                  const ITensorInfo *output, int axis)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, indices, output);
  ARM_COMPUTE_RETURN_ERROR_ON(indices->num_dimensions() > 3);
  ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
  ARM_COMPUTE_ERROR_ON(input->num_dimensions() + indices->num_dimensions() - 1 > 4);

  if (axis < 0)
  {
    axis += input->num_dimensions();
  }

  ARM_COMPUTE_RETURN_ERROR_ON(0 > axis || axis >= static_cast<int32_t>(input->num_dimensions()));
  ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(
      input, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::U16, DataType::S16,
      DataType::U32, DataType::S32, DataType::F16, DataType::F32);

  if (output->total_size() != 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    TensorShape output_shape = arm_compute::misc::shape_calculator::compute_gather_shape_ex(
        input->tensor_shape(), indices->tensor_shape(), axis);
    ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() != output->tensor_shape().total_size());
  }

  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32, DataType::S32);

  return Status{};
}

void NEGatherKernelEx::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON(_func == nullptr);

  (this->*_func)(window, info);
}

} // namespace arm_compute
