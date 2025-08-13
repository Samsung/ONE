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

// ------------------------------------------------------------------------------------------------

template <typename T, template <typename> class ReduceFn>
bool Reducer<T, ReduceFn>::ReduceImpl(bool mean)
{
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
T Reducer<T, ReduceFn>::ResolvedAxisLength()
{
  T axis_length = 1;
  constexpr static auto kMax = std::numeric_limits<size_t>::max();

  for (auto i = 0u; i < _resolved_axis.size(); ++i)
  {
    auto& axis = _resolved_axis.at(i);
    auto current = static_cast<size_t>(_input_ctx.Dims()[axis]);

    if (current == 0)
      return false;

    // Overflow prevention.
    if (current > (kMax / axis_length))
      return false;

    axis_length *= current;
  }

  return axis_length;
}

} // namespace onert_micro::execute::pal

#endif // ONERT_MICRO_PAL_REDUCE_COMMON_H
