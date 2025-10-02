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

#include <set>
#include <unordered_map>

namespace core = onert_micro::core;

using core::OMRuntimeShape;
using core::OMReduceDataContext;
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

template <typename T>
struct ReduceMaxFn
{
  void operator()(T& total, const T value)
  {
    total = std::max(total, value);
  }
};

// ------------------------------------------------------------------------------------------------

template <typename T, template <typename> class ReduceFn>
class Reducer
{
  using ValueType = std::conditional_t<IsQuantized<T>, float, T>;

private:
  core::OMInputContext<T> &_input;
  core::OMOutputContext<T> &_output;
  core::OMAxisContext &_axes;

  T _init_value;
  ReduceFn<ValueType> _reducer;

  std::unordered_map<size_t, size_t> _curr_index = {};
  std::unordered_map<size_t, uint32_t> _resolved_axes = {};
  std::unordered_map<size_t, ValueType> _accumulator = {};

public:
  explicit Reducer(core::OMReduceDataContext<T> &ctx, T init_value)
    : _input(ctx.Input())
    , _output(ctx.Output())
    , _axes(ctx.Axis())
    , _init_value(init_value)
  {}

public:
  bool Mean()
  {
    if (SpecialCaseMeanImpl())
      return true;

    return ReduceImpl(true);
  }

  bool Reduce()
  {
    return ReduceImpl();
  }

private:
  bool ReduceImpl(bool mean = false);
  bool SpecialCaseMeanImpl();

  bool ResolveAxis();
  T ResolvedAxisLength();

  size_t ReducedOutputOffset(int num_axes, const uint32_t *axes);
  bool NextIndex();
};

// ------------------------------------------------------------------------------------------------

template <typename T, template <typename> class ReduceFn>
bool Reducer<T, ReduceFn>::ResolveAxis()
{
  size_t num_resolved_axes = 0;
  _resolved_axes.clear();

  if (_input.IsScalar())
    return 0;

  for (size_t i = 0; i < _axes.ElementsCount(); ++i)
  {
    int current = _axes.Data().At(i);

    if (_resolved_axes.count(current) > 0)
      continue;

    if (_resolved_axes.size() > 1)
      return false;

    _resolved_axes[num_resolved_axes++] = current;
  }

  return true;
}

template <typename T, template <typename> class ReduceFn>
bool Reducer<T, ReduceFn>::SpecialCaseMeanImpl()
{
  /*
    Case: Mean over WH of axis 1 and 2.
    Detail: for tensor with rank=4 and simultaneous reduction over width and height.
  */
  const uint32_t *axes_data = _axes.Data().Get();
  std::set<uint32_t> axes_values = { axes_data[0], axes_data[1] };

  if (_input.DimsCount() != 4)
    return false;

  if (_axes.ElementsCount() != 2)
    return false;

  if (axes_values.count(1) != 1 || axes_values.count(2) != 1)
    return false;

  auto input_shape = OMRuntimeShape::extendedShape(4, _input.Shape());
  auto output_shape = OMRuntimeShape::extendedShape(4, _output.Shape());

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
          size_t idx = offset(input_shape.dimsData(), out_b, in_h, in_w, out_d);
          value += static_cast<float>(_input.Data().At(idx));
        }
      }

      float result = value / (input_width * input_height);
      size_t idx = offset(output_shape.dimsData(), out_b, 0, 0, out_d);
      _output.Data().SetAt(idx, result);
    }
  }

  return true;
}

template <typename T, template <typename> class ReduceFn>
bool Reducer<T, ReduceFn>::ReduceImpl(bool mean)
{
  _accumulator.clear();
  _curr_index.clear();

  auto *axes_data = _axes.Data().Get();
  auto num_outputs = _output.ElementsCount();

  const auto &input_data = _input.Data();

  if (_input.HasZeroSizeDims())
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

  do
  {
    size_t input_offset = ReducedOutputOffset(0, nullptr);
    size_t output_offset = ReducedOutputOffset(_resolved_axes.size(), axes_data);

    _reducer(_accumulator[output_offset], input_data.ValueAt(input_offset));

  } while (NextIndex());

  for (size_t i = 0; i < num_outputs; ++i)
  {
    auto value = _accumulator.at(i);

    if (mean)
    {
      value /= ResolvedAxisLength();
    }

    _output.Data().SetValueAt(i, value);
  }

  return true;
}

template <typename T, template <typename> class ReduceFn>
T Reducer<T, ReduceFn>::ResolvedAxisLength()
{
  T axis_length = 1;
  constexpr static auto kMax = std::numeric_limits<size_t>::max();

  for (auto i = 0u; i < _resolved_axes.size(); ++i)
  {
    auto &axis = _resolved_axes.at(i);
    auto current = static_cast<size_t>(_input.Dims()[axis]);

    if (current == 0)
      return false;

    // Overflow prevention.
    if (current > (kMax / axis_length))
      return false;

    axis_length *= current;
  }

  return static_cast<T>(axis_length);
}

/*
  Gets offset of index if reducing on axis. When reducing, the flattened offset
  will not change, if the input index changes on the given axis.
  For example, if you have a 3D tensor and you are reducing to 2D by eliminating axis 0,
  then index (0, 1, 2) and index (1, 1, 2) will map to the same flattened offset.
*/
template <typename T, template <typename> class ReduceFn>
size_t Reducer<T, ReduceFn>::ReducedOutputOffset(int num_axes, const uint32_t *axes_data)
{
  size_t offset = 0;

  for (auto dim_idx = 0u; dim_idx < _input.DimsCount(); ++dim_idx)
  {
    bool skip_axis = false;

    if (axes_data != nullptr)
    {
      skip_axis = std::any_of(axes_data, axes_data + num_axes, [&dim_idx](auto axis)
      {
        return axis == dim_idx;
      });
    }

    if (!skip_axis)
    {
      offset *= _input.DimLength(dim_idx);
      offset += _curr_index[dim_idx];
    }
  }

  return offset;
}

/*
  Gets next index to iterate through a multidimensional array.
*/
template <typename T, template <typename> class ReduceFn>
bool Reducer<T, ReduceFn>::NextIndex()
{
  if (_input.DimsCount() == 0)
  {
    return false;
  }

  for (int idx = _input.DimsCount() - 1; idx >= 0; --idx)
  {
    auto current_val = _curr_index[idx] + 1;

    if (_input.DimLength(idx) != current_val)
    {
      _curr_index[idx] = current_val;
      return true;
    }

    _curr_index[idx] = 0;
  }

  return false;
}

// ------------------------------------------------------------------------------------------------

} // namespace onert_micro::execute::pal

#endif // ONERT_MICRO_PAL_REDUCE_COMMON_H
