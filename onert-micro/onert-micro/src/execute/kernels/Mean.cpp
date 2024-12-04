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

#include "OMStatus.h"

#include "core/OMUtils.h"
#include "core/OMKernelData.h"

#include "execute/OMKernelExecutionBuilder.h"
#include "execute/OMRuntimeKernel.h"

#include "PALMean.h"

using namespace onert_micro;
using namespace onert_micro::core;
using namespace onert_micro::execute;

namespace
{

constexpr uint32_t input1TensorIdx = 0;
constexpr uint32_t input2TensorIdx = 1;
constexpr uint32_t outputTensorIdx = 0;

const int kMaxNumberOfAxis = 5;
const int kMaxNumberOfReducedAxis = 2;

void resolveAxis(const int32_t *axis_data, int32_t axis_count, core::MeanParams &op_params)
{
  int32_t i = 0;
  for (; i < axis_count; ++i)
  {
    op_params.axis[i] = static_cast<int16_t>(axis_data[i]);
  }
  for (; i < maxTensorShapeSize; ++i)
  {
    op_params.axis[i] = 1;
  }
  op_params.axis_count = axis_count;
}

} // namespace

OMStatus onert_micro::execute::execute_kernel_CircleMean(const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &runtime_context = execute_args.runtime_context;
  core::OMRuntimeStorage &runtime_storage = execute_args.runtime_storage;
  uint16_t op_index = execute_args.kernel_index;

  const circle::Tensor *input;
  const circle::Tensor *axis;
  const circle::Tensor *output;

  uint8_t *input_data;
  uint8_t *axis_data;
  uint8_t *output_data;

  // Read kernel
  const circle::ReducerOptions *options;

  core::MeanParams params{};
  {
    execute::OMRuntimeKernel runtime_kernel;
    OMStatus status = runtime_kernel.readKernel(op_index, runtime_context);
    if (status != Ok)
      return status;

    input = runtime_kernel.inputs[input1TensorIdx];
    axis = runtime_kernel.inputs[input2TensorIdx];
    output = runtime_kernel.outputs[outputTensorIdx];
    assert(input != nullptr);
    assert(axis != nullptr);
    assert(output != nullptr);

    status = runtime_kernel.getDataFromStorage(op_index, runtime_storage, runtime_context);
    if (status != Ok)
      return status;

    input_data = runtime_kernel.inputs_data[input1TensorIdx];
    axis_data = runtime_kernel.inputs_data[input2TensorIdx];
    output_data = runtime_kernel.outputs_data[outputTensorIdx];
    assert(input_data != nullptr);
    assert(axis_data != nullptr);
    assert(output_data != nullptr);
    options = runtime_kernel.first_operator->builtin_options_as_ReducerOptions();
  }
  OMStatus status = Ok;
  OMRuntimeShape axis_shape(axis);
  OMRuntimeShape input_shape(input);
  OMRuntimeShape output_shape(output);

  auto *axis_value = reinterpret_cast<int32_t *>(axis_data);
  for (int32_t i = 0; i < axis_shape.flatSize(); ++i)
  {
    if (axis_value[i] < 0)
    {
      axis_value[i] += input_shape.dimensionsCount() + 1;
    }
  }

  resolveAxis(axis_value, axis_shape.flatSize(), params);

  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
    {
      // Special case mean implementation exists for 4D mean across axes 1
      // and 2.
      bool special_case_4d_axes_1_and_2 = input_shape.dimensionsCount() == 4 &&
                                          params.axis_count == 2 &&
                                          ((params.axis[0] == 1 && params.axis[1] == 2) ||
                                           (params.axis[0] == 2 && params.axis[1] == 1));

      // Defer to specialized implementation for 4D Mean across axes 1 & 2.
      if (options->keep_dims() && special_case_4d_axes_1_and_2)
      {
        status = pal::Mean(params, input_shape, utils::castInputData<float>(input_data),
                           output_shape, utils::castOutputData<float>(output_data));
      }
      else
      {
        int temp_index[kMaxNumberOfAxis];
        int resolved_axis[kMaxNumberOfReducedAxis];
        status = pal::Mean(utils::castInputData<float>(input_data), input_shape,
                           utils::castOutputData<float>(output_data), output_shape, axis_value,
                           axis_shape.flatSize(), options->keep_dims(), temp_index, resolved_axis,
                           utils::castOutputData<float>(output_data));
      }
    }
    break;
#endif // DIS_FLOAT
    default:
    {
      status = UnsupportedActivation;
      assert(false && "Unsupported type.");
      break;
    }
  }

  return status;
}
