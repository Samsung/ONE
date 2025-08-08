/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#ifndef ONERT_MICRO_PAL_REDUCE_COMMON_H
#define ONERT_MICRO_PAL_REDUCE_COMMON_H

#include "PALUtils.h"
#include "core/OMCustomRuntimeData.h"
#include "core/OMRuntimeShape.h"
#include "core/OMTypeTraits.h"

#include <unordered_map>

namespace core = onert_micro::core;

using core::type_traits::IsQuantized;

namespace onert_micro::execute::pal
{

// clang-format off

// ------------------------------------------------------------------------------------------------

template <class T>
struct ReduceSumFn
{
  void operator()(T& total, const T value)
  {
    total += value;
  }
};

// ------------------------------------------------------------------------------------------------

template <typename T>
struct ReduceProductFn
{
  void operator()(T& total, const T value)
  {
    total *= value;
  }
};

// ------------------------------------------------------------------------------------------------

template <typename T, template <typename> class ReduceFn>
class Reducer
{
  using ValueType = std::conditional_t<IsQuantized<T>, float, T>;

  core::OMInputContext<T> &_input_ctx;
  core::OMOutputContext<T> &_output_ctx;
  core::OMAxisContext<1> &_axis_ctx;

  T _init_value;
  ReduceFn<ValueType> _reducer;

  std::unordered_map<size_t, int> _resolved_axis = {};
  std::unordered_map<size_t, ValueType> _accumulator = {};

public:
  explicit Reducer(core::OMReduceDataContext<T> &ctx, T init_value)
    : _input_ctx(ctx.Input())
    , _output_ctx(ctx.Output())
    , _axis_ctx(ctx.Axis())
    , _init_value(init_value)
  {}

public:
  bool Mean()
  {
    return ReduceImpl(true);
  }

  bool Reduce()
  {
    return ReduceImpl();
  }

private:
  size_t IndexOffset(const int32_t *index, const int32_t num_axis, const int32_t *axis)
  {
    return reducedOutputOffset(_input_ctx.DimsCount(), _input_ctx.Dims(), index, num_axis, axis);
  }

private:
  bool ReduceImpl(bool mean = false);

  bool ResolveAxis();
  T ResolvedAxisLength();
};

// ------------------------------------------------------------------------------------------------

template <typename T, template <typename> class ReduceFn>
bool Reducer<T, ReduceFn>::ResolveAxis()
{
  size_t num_resolved_axis = 0;
  _resolved_axis.clear();

  if (_input_ctx.IsScalar())
    return 0;

  for (size_t i = 0; i < _axis_ctx.DimsCount(); ++i)
  {
    int current = _axis_ctx.Data().PositiveAxisAt(i);

    if (_resolved_axis.count(current) > 0)
      continue;

    if (_resolved_axis.size() > 1)
      return false;

    _resolved_axis[num_resolved_axis++] = current;
  }

  return true;
}

// Mean over WH of axis 1,2
template <typename T>
inline void MeanROWH(const OMRuntimeShape &unextended_input_shape, const T *input_data,
                     const OMRuntimeShape &unextended_output_shape, T *output_data)
{
  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  const OMRuntimeShape input_shape = OMRuntimeShape::extendedShape(4, unextended_input_shape);
  const OMRuntimeShape output_shape = OMRuntimeShape::extendedShape(4, unextended_output_shape);

  const int output_batch = output_shape.dims(0);
  const int output_depth = output_shape.dims(3);

  const int input_height = input_shape.dims(1);
  const int input_width = input_shape.dims(2);

  for (int out_b = 0; out_b < output_batch; ++out_b)
  {
    for (int out_d = 0; out_d < output_depth; ++out_d)
    {
      float value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h)
      {
        for (int in_w = 0; in_w < input_width; ++in_w)
        {
          value += static_cast<float>(
            input_data[offset(input_shape.dimsData(), out_b, in_h, in_w, out_d)]);
        }
      }
      float result = value / (input_width * input_height);
      output_data[offset(output_shape.dimsData(), out_b, 0, 0, out_d)] = static_cast<T>(result);
    }
  }
}

template <typename T, template <typename> class ReduceFn>
bool Reducer<T, ReduceFn>::ReduceImpl(bool mean)
{
  // Special case mean implementation exists for 4D mean across axes 1 and 2

  const int *axis_value = ctx.Axis().Data().Get();

  bool special_case_4d_axes_1_and_2 =
    ctx.Input().DimsCount() == 4 && ctx.Axis().ShapeFlatSize() == 2 &&
    ((axis_value[0] == 1 && axis_value[1] == 2) || (axis_value[0] == 2 && axis_value[1] == 1));

  if (special_case_4d_axes_1_and_2)
  {
    OMRuntimeShape input_shape(ctx.Input().Shape());
    OMRuntimeShape output_shape(ctx.Output().Shape());
    MeanROWH<T>(input_shape, ctx.Input().Data().Get(), output_shape, ctx.Output().Data().Get());
    return true;
  }

  auto &input_data = _input_ctx.Data();
  auto input_dims = _input_ctx.Dims();
  auto input_num_dims = _input_ctx.DimsCount();

  auto *axis_data = _axis_ctx.Data().Get();

  auto &output_data = _output_ctx.Data();
  auto num_outputs = _output_ctx.ShapeFlatSize();

  if (_input_ctx.HasZeroSizeDims())
  {
    return false;
  }

  for (size_t i = 0; i < num_outputs; ++i)
  {
    _accumulator[i] = _init_value;
  }

  if (!ResolveAxis())
  {
    return false;
  }

  int temp_index[5] = {0};
  do
  {
    size_t input_offset = IndexOffset(temp_index, 0, nullptr);
    size_t output_offset = IndexOffset(temp_index, _resolved_axis.size(), axis_data);

    _reducer(_accumulator.at(output_offset), input_data.ValueAt(input_offset));

  } while (nextIndex(input_num_dims, input_dims, temp_index));

  for (size_t i = 0; i < num_outputs; ++i)
  {
    auto value = _accumulator.at(i);

    if (mean)
    {
      value /= ResolvedAxisLength();
    }

    output_data.SetValueAt(i, value);
  }

  return true;
}

// ------------------------------------------------------------------------------------------------

template <typename T, template <typename> class ReduceFn>
bool Reduce(OMReduceDataContext<T> &ctx, bool mean = false)
{
  // Special case mean implementation exists for 4D mean across axes 1
  // and 2
  const int *axis_value = ctx.Axis().Data().Get();
  bool special_case_4d_axes_1_and_2 =
    ctx.Input().DimsCount() == 4 && ctx.Axis().ShapeFlatSize() == 2 &&
    ((axis_value[0] == 1 && axis_value[1] == 2) || (axis_value[0] == 2 && axis_value[1] == 1));
  if (special_case_4d_axes_1_and_2)
  {
    OMRuntimeShape input_shape(ctx.Input().Shape());
    OMRuntimeShape output_shape(ctx.Output().Shape());
    MeanROWH<T>(input_shape, ctx.Input().Data().Get(), output_shape, ctx.Output().Data().Get());
    return true;
  }

  constexpr static T kInitValue = T(0);

  if (!ReduceGeneric<T, ReduceFn>(ctx))
  {
    return false;
  }

  auto &input = ctx.Input();
  auto input_dims = input.Dims();
  auto input_num_dims = input.DimsCount();

  auto &output = ctx.Output();
  auto &output_data = output.Data();
  auto num_outputs = output.ShapeFlatSize();

  auto &axis = ctx.Axis().Data();
  auto num_axis_dimensions = ctx.Axis().DimsCount();

  // Resolve axis again for computing mean
  int num_resolved_axis = 0;
  int resolved_axis[2];

  if (!resolveAxis(input_num_dims, axis.Get(), num_axis_dimensions, resolved_axis,
                   &num_resolved_axis))
  {
    return false;
  }

  // clang-format off

  auto fnReduceOutput = [&](size_t divide_by = 1)
  {
      for (size_t idx = 0; idx < num_outputs; ++idx)
      {
        auto value = output_data.ValueAt(idx);
        value /= static_cast<T>(divide_by);
        output_data.SetAt(idx, value);
      }
  };

  // clang-format on

  if (!mean)
  {
    fnReduceOutput();
    return true;
  }

  // Calculate mean by dividing output_data by num of aggregated element.
  size_t num_elements_in_axis = 1;
  for (int idx = 0; idx < num_resolved_axis; ++idx)
  {
    size_t current = static_cast<size_t>(input_dims[resolved_axis[idx]]);
    // Overflow prevention.
    if (current > (kMax / axis_length))
      return false;

    axis_length *= current;
  }

  return axis_length;
}

} // namespace onert_micro::execute::pal

#endif // ONERT_MICRO_PAL_REDUCE_COMMON_H
