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
#include "core/OMUtils.h"
#include "PALFloorMod.h"

using namespace onert_micro;
using namespace onert_micro::core;

namespace
{

constexpr uint32_t input1TensorIdx = 0;
constexpr uint32_t input2TensorIdx = 1;
constexpr uint32_t outputTensorIdx = 0;

} // namespace

// NOTE: doesnt currently support dynamic shapes
OMStatus onert_micro::execute::execute_kernel_CircleFloorMod(const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &runtime_context = execute_args.runtime_context;
  core::OMRuntimeStorage &runtime_storage = execute_args.runtime_storage;
  uint16_t op_index = execute_args.kernel_index;

  const circle::Tensor *input1 = nullptr;
  const circle::Tensor *input2 = nullptr;
  const circle::Tensor *output = nullptr;

  uint8_t *input_data1 = nullptr;
  uint8_t *input_data2 = nullptr;
  uint8_t *output_data = nullptr;

  OMStatus status = Ok;

  {
    OMRuntimeKernel runtime_kernel;
    runtime_kernel.readKernel(op_index, runtime_context);

    input1 = runtime_kernel.inputs[input1TensorIdx];
    input2 = runtime_kernel.inputs[input2TensorIdx];
    output = runtime_kernel.outputs[outputTensorIdx];

    assert(input1 != nullptr);
    assert(input2 != nullptr);
    assert(output != nullptr);

    status = runtime_kernel.getDataFromStorage(op_index, runtime_storage, runtime_context);
    if (status != Ok)
      return status;

    input_data1 = runtime_kernel.inputs_data[input1TensorIdx];
    input_data2 = runtime_kernel.inputs_data[input2TensorIdx];
    output_data = runtime_kernel.outputs_data[outputTensorIdx];
  }

  assert(input_data1 != nullptr);
  assert(input_data2 != nullptr);
  assert(output_data != nullptr);

  core::OMRuntimeShape input1_shape(input1);
  core::OMRuntimeShape input2_shape(input2);
  core::OMRuntimeShape output_shape(output);

  switch (input1->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
    {
      // Check the denominator
      for (int i = 0; i < input2_shape.flatSize(); ++i)
      {
        utils::checkCondition(core::utils::castInputData<float>(input_data2)[i] != 0);
      }
      // check that input and output dimensions are equal
      if (input1_shape == input2_shape)
      {
        const int flat_size = input1_shape.flatSize();
        pal::FloorMod(flat_size, core::utils::castInputData<float>(input_data1),
                      core::utils::castInputData<float>(input_data2),
                      core::utils::castOutputData<float>(output_data));
      }
      else
      {
        pal::BroadcastFloorMod4DSlow(input1_shape, core::utils::castInputData<float>(input_data1),
                                     input2_shape, core::utils::castInputData<float>(input_data2),
                                     output_shape, core::utils::castOutputData<float>(output_data));
      }
    }
    break;
#endif // DIS_FLOAT
    default:
      assert(false && "Unsupported type.");
  }

  return status;
}
