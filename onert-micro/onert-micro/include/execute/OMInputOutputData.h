/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_EXECUTE_INPUT_OUTPUT_DATA_H
#define ONERT_MICRO_EXECUTE_INPUT_OUTPUT_DATA_H

#include "core/OMRuntimeShape.h"
#include "core/OMUtils.h"
#include "execute/OMRuntimeKernel.h"

#include <cstdint>

namespace onert_micro::execute
{

// ------------------------------------------------------------------------------------------------

template <class T> class OMInputOutputData
{
protected:
  const circle::Tensor *_input;
  const circle::Tensor *_output;

  const T *_input_data;
  T *_output_data;

  core::OMRuntimeShape _input_shape;
  core::OMRuntimeShape _output_shape;

protected:
  constexpr static int32_t kInputTensorIdx = 0;
  constexpr static int32_t kOutputTensorIdx = 0;

public:
  explicit OMInputOutputData(const OMRuntimeKernel &rt_kernel)
  {
    using namespace onert_micro::core::utils;

    _input = rt_kernel.inputs[kInputTensorIdx];
    _output = rt_kernel.outputs[kOutputTensorIdx];

    _input_shape = _input;
    _output_shape = _output;

    assert(_input != nullptr);
    assert(_output != nullptr);

    _input_data = castInputData<T>(rt_kernel.inputs_data[kInputTensorIdx]);
    _output_data = castOutputData<T>(rt_kernel.outputs_data[kOutputTensorIdx]);

    assert(_input_data != nullptr);
    assert(_output_data != nullptr);
  }

public:
  const circle::Tensor *Input() const { return _input; }
  const circle::Tensor *Output() const { return _output; }

  const T *InputData() const { return _input_data; }
  T *OutputData() { return _output_data; }

  const core::OMRuntimeShape &InputShape() const { return _input_shape; }
  const core::OMRuntimeShape &OutputShape() const { return _output_shape; }
};

// ------------------------------------------------------------------------------------------------

template <uint32_t AxisTensorIndex> class OMAxisData
{
protected:
  const circle::Tensor *_axis;
  const int32_t *_axis_data;
  core::OMRuntimeShape _axis_shape;

protected:
  constexpr static uint32_t kIndex = AxisTensorIndex;

public:
  explicit OMAxisData(const OMRuntimeKernel &rt_kernel)
  {
    using namespace onert_micro::core::utils;

    _axis = rt_kernel.inputs[kIndex];
    _axis_shape = _axis;
    _axis_data = castInputData<int32_t>(rt_kernel.inputs_data[kIndex]);

    assert(_axis != nullptr);
    assert(_axis_data != nullptr);
  }

public:
  const circle::Tensor *AxisTensor() const { return _axis; }

  const int32_t *AxisData() const { return _axis_data; }

  const core::OMRuntimeShape &AxisShape() const { return _axis_shape; }
};

// ------------------------------------------------------------------------------------------------

} // namespace onert_micro::execute

#endif // ONERT_MICRO_EXECUTE_INPUT_OUTPUT_DATA_H
