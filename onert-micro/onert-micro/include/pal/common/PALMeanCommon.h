/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_PAL_MEAN_COMMON_H
#define ONERT_MICRO_PAL_MEAN_COMMON_H

#include "execute/OMInputOutputData.h"
#include "PALReduceCommon.h"

namespace onert_micro::execute::pal
{

template <typename T> bool Mean(OMInputOutputData<T> &io_data, const OMAxisData<1> &axis_data)
{
  using namespace onert_micro::execute::pal;

  constexpr static T kInitValue = T(0);

  auto is_ok = ReduceGeneric<T, ReduceSumFn<T>>(io_data, axis_data, kInitValue);
  if (!is_ok)
  {
    return false;
  }

  // Resolve axis again for computing mean

  const int *input_dims = io_data.InputShape().dimsData();
  int input_num_dims = io_data.InputShape().dimensionsCount();

  const int *axis = axis_data.AxisData();
  int num_axis_dimensions = axis_data.AxisShape().dimensionsCount();

  int num_resolved_axis = 0;
  int resolved_axis[2];

  if (!resolveAxis(input_num_dims, axis, num_axis_dimensions, resolved_axis, &num_resolved_axis))
  {
    return false;
  }

  // Calculate mean by dividing output_data by num of aggregated element.

  size_t num_elements_in_axis = 1;

  for (int idx = 0; idx < num_resolved_axis; ++idx)
  {
    size_t current = static_cast<size_t>(input_dims[resolved_axis[idx]]);

    if (current > (std::numeric_limits<size_t>::max() / num_elements_in_axis))
    {
      // Overflow prevention.
      return false;
    }

    num_elements_in_axis *= current;
  }

  if (num_elements_in_axis == 0)
  {
    return true;
  }

  T *output_data = io_data.OutputData();
  size_t num_outputs = io_data.OutputShape().flatSize();

  for (size_t idx = 0; idx < num_outputs; ++idx)
  {
    output_data[idx] = static_cast<T>(output_data[idx] / static_cast<T>(num_elements_in_axis));
  }

  return true;
}

} // namespace onert_micro::execute::pal

#endif // ONERT_MICRO_PAL_MEAN_COMMON_H
