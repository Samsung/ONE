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

#include "core/OMRuntimeShape.h"
#include "core/OMUtils.h"

#include "execute/OMKernelExecutionBuilder.h"
#include "execute/OMRuntimeKernel.h"
#include "execute/OMUtils.h"

#include "PALReduceCommon.h"

using namespace onert_micro;
using namespace onert_micro::execute;

// ------------------------------------------------------------------------------------------------

#if 0 // TODO: Remove unused?
namespace
{

template <typename T>
void reduceMeanGeneric(core::OMRuntimeShape &input_shape, const T *input_data,
                       core::OMRuntimeShape &axis_shape, const int *axis_data,
                       core::OMRuntimeShape &output_shape, T *output_data, bool keep_dims)
{
  onert_micro::execute::pal::ReduceGeneric<T>(
    input_data, input_shape.dimsData(), input_shape.dimensionsCount(), output_data, axis_data,
    axis_shape.dimensionsCount(),
    /*init_value=*/T(0), output_shape.flatSize(),
    [](const T current, const T in) -> T { return in + current; });
}

} // namespace
#endif

// ------------------------------------------------------------------------------------------------

namespace onert_micro::execute
{

OMStatus execute_kernel_CircleMean(const OMExecuteArgs &execute_args)
{
  constexpr static uint32_t inputTensorIdx = 0;
  constexpr static uint32_t axisTensorIdx = 1;
  constexpr static uint32_t outputTensorIdx = 0;

  core::OMRuntimeContext &runtime_context = execute_args.runtime_context;
  core::OMRuntimeStorage &runtime_storage = execute_args.runtime_storage;

  const uint16_t op_index = execute_args.kernel_index;

  execute::OMRuntimeKernel runtime_kernel;
  runtime_kernel.readKernel(op_index, runtime_context);

  const circle::Tensor *input = runtime_kernel.inputs[inputTensorIdx];
  const circle::Tensor *axis = runtime_kernel.inputs[axisTensorIdx];
  const circle::Tensor *output = runtime_kernel.outputs[outputTensorIdx];

  assert(input != nullptr);
  assert(axis != nullptr);
  assert(output != nullptr);

  runtime_kernel.getDataFromStorage(op_index, runtime_storage, runtime_context);

  uint8_t *input_data = runtime_kernel.inputs_data[inputTensorIdx];
  uint8_t *axis_data = runtime_kernel.inputs_data[axisTensorIdx];
  uint8_t *output_data = runtime_kernel.outputs_data[outputTensorIdx];

  assert(input_data != nullptr);
  assert(axis_data != nullptr);
  assert(output_data != nullptr);

  // TODO: Remove unused?
  // uint16_t input_index = runtime_kernel.inputs_index[inputTensorIdx];
  // uint16_t axis_index = runtime_kernel.inputs_index[axisTensorIdx];
  // auto options = runtime_kernel.first_operator->builtin_options_as_ReducerOptions();

  const core::OMRuntimeShape input_shape(input);
  const core::OMRuntimeShape axis_shape(axis);
  const core::OMRuntimeShape output_shape(output);

  // clang-format off

  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
    {
      const bool is_ok = pal::Mean<float>
      (
        input_shape.dimsData(), core::utils::castInputData<float>(input_data),
        input_shape.dimensionsCount(), core::utils::castOutputData<float>(output_data),
        output_shape.flatSize(), core::utils::castInputData<int>(axis_data),
        axis_shape.dimensionsCount()
      );

      return is_ok ? Ok : UnknownError;
    }
#endif // DIS_FLOAT

    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
    default:
      assert(false && "Unsupported type");
      break;
  }

  return UnsupportedType;
}

} // namespace onert_micro::execute
