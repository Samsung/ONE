/* Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "luci_compute/Types.h"
#include "luci_compute/StridedSlice.h"

#include "ConvertTypes.h"
#include "ConvertValues.h"

#include <tensorflow/lite/kernels/internal/reference/strided_slice.h>

#include <cassert>
#include <cstdint>

namespace luci
{
namespace compute
{

template bool StridedSlice<float>::prepare(void);
template bool StridedSlice<int32_t>::prepare(void);
template void StridedSlice<float>::compute(void);
template void StridedSlice<int32_t>::compute(void);

template <typename T> bool StridedSlice<T>::prepare(void)
{
  assert(_begin_shape.rank() == 1);
  assert(_end_shape.rank() == 1);
  assert(_strides_shape.rank() == 1);
  assert(_input_shape.rank() <= 4);
  if (_params.ellipsis_mask != 0)
  {
    throw std::runtime_error("ellipsis_mask is not implemented yet.");
  }
  if (_params.new_axis_mask != 0)
  {
    throw std::runtime_error("new_axis_mask is not implemented yet.");
  }

  tflite::StridedSliceParams params;

  // clang-format off
  params.start_indices_count   = _params.start_indices_count;
  params.stop_indices_count    = _params.stop_indices_count;
  params.strides_count         = _params.strides_count;
  for (auto i = 0; i < _input_shape.rank(); ++i)
  {
    params.start_indices[i]    = _params.start_indices[i];
    params.stop_indices[i]     = _params.stop_indices[i];
    params.strides[i]          = _params.strides[i];
  }
  params.begin_mask            = _params.begin_mask;
  params.ellipsis_mask         = 0;
  params.end_mask              = _params.end_mask;
  params.new_axis_mask         = 0;
  params.shrink_axis_mask      = _params.shrink_axis_mask;
  // clang-format on

  std::vector<int32_t> output_shape_vector;
  for (auto i = 0; i < _input_shape.rank(); ++i)
  {
    auto idx = _input_shape.rank() - i - 1;
    auto stride = _strides_data[idx];
    assert(stride != 0);
    auto begin = ::tflite::strided_slice::StartForAxis(params, tflite_shape(_input_shape), idx);
    auto end = ::tflite::strided_slice::StopForAxis(params, tflite_shape(_input_shape), idx, begin);

    const bool shrink_axis = params.shrink_axis_mask & (1 << idx);
    if (shrink_axis)
    {
      end = begin + 1;
    }

    auto dim_shape = std::ceil((end - begin) / static_cast<float>(stride));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    if (!shrink_axis)
    {
      output_shape_vector.emplace_back(dim_shape);
    }
  }

  _output_shape.rank(output_shape_vector.size());
  for (auto i = 0; i < output_shape_vector.size(); ++i)
  {
    _output_shape.dim(i) = output_shape_vector[output_shape_vector.size() - i - 1];
  }

  return true;
}

template <typename T> void StridedSlice<T>::compute(void)
{
  // NOTE if this fails, structure may have changed
  static_assert(sizeof(compute::StridedSliceParams) == sizeof(tflite::StridedSliceParams));

  tflite::StridedSliceParams params;

  // clang-format off
  params.start_indices_count   = _params.start_indices_count;
  params.stop_indices_count    = _params.stop_indices_count;
  params.strides_count         = _params.strides_count;
  for (int i = 0; i < _input_shape.rank(); i++)
  {
    params.start_indices[i]    = _params.start_indices[i];
    params.stop_indices[i]     = _params.stop_indices[i];
    params.strides[i]          = _params.strides[i];
  }
  params.begin_mask            = _params.begin_mask;
  params.ellipsis_mask         = _params.ellipsis_mask;
  params.end_mask              = _params.end_mask;
  params.new_axis_mask         = _params.new_axis_mask;
  params.shrink_axis_mask      = _params.shrink_axis_mask;
  // clang-format on

  tflite::reference_ops::StridedSlice(params, tflite_shape(_input_shape), _input_data,
                                      tflite_shape(_output_shape), _output_data);
}

} // namespace compute
} // namespace luci
