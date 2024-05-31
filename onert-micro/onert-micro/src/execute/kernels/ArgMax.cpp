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

#include "execute/OMKernelExecutionBuilder.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"

#include "core/OMRuntimeShape.h"
#include "PALArgMax.h"

using namespace onert_micro;
using namespace onert_micro::execute;

namespace
{
constexpr uint32_t input1TensorIdx = 0;
constexpr uint32_t input2TensorIdx = 1;
constexpr uint32_t outputTensorIdx = 0;
} // namespace

OMStatus onert_micro::execute::execute_kernel_CircleArgMax(const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &runtime_context = execute_args.runtime_context;
  core::OMRuntimeStorage &runtime_storage = execute_args.runtime_storage;
  uint16_t op_index = execute_args.kernel_index;
  const circle::Tensor *output;
  const circle::Tensor *input1;
  const circle::Tensor *input2;

  uint8_t *output_data;
  uint8_t *input_data;
  uint8_t *axis_data;

  // Read kernel
  execute::OMRuntimeKernel runtime_kernel;
  runtime_kernel.readKernel(op_index, runtime_context);

  output = runtime_kernel.outputs[outputTensorIdx];
  assert(output != nullptr);

  input1 = runtime_kernel.inputs[input1TensorIdx];
  assert(input1 != nullptr);

  input2 = runtime_kernel.inputs[input2TensorIdx];
  assert(input2 != nullptr);

  runtime_kernel.getDataFromStorage(op_index, runtime_storage, runtime_context);

  output_data = runtime_kernel.outputs_data[outputTensorIdx];
  assert(output_data != nullptr);

  input_data = runtime_kernel.inputs_data[input1TensorIdx];
  assert(input_data != nullptr);

  axis_data = runtime_kernel.inputs_data[input2TensorIdx];
  assert(axis_data != nullptr);

  OMStatus status;
  const core::OMRuntimeShape input1_shape(input1);
  const core::OMRuntimeShape output_shape(output);
  switch (input1->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
    {
      status =
        onert_micro::execute::pal::ArgMax(input1_shape, reinterpret_cast<const float *>(input_data),
                                          reinterpret_cast<const int *>(axis_data), output_shape,
                                          reinterpret_cast<int *>(output_data));
    }
    break;
#endif // DIS_FLOAT
    default:
    {
      status = UnsupportedType;
      assert(false && "Unsupported type.");
    }
  }
  return status;
}
