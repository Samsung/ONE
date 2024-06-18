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
#include "PALDequantize.h"
#include "core/OMRuntimeShape.h"

using namespace onert_micro;
using namespace onert_micro::execute;

namespace
{

constexpr uint32_t inputTensorIdx = 0;
constexpr uint32_t outputTensorIdx = 0;

} // namespace

// NOTE: doesnt currently support dynamic shapes
OMStatus onert_micro::execute::execute_kernel_CircleDequantize(const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &runtime_context = execute_args.runtime_context;
  core::OMRuntimeStorage &runtime_storage = execute_args.runtime_storage;
  uint16_t op_index = execute_args.kernel_index;

  const circle::Tensor *input = nullptr;
  const circle::Tensor *output = nullptr;

  uint8_t *input_data = nullptr;
  uint8_t *output_data = nullptr;

  OMStatus status = Ok;

  {
    OMRuntimeKernel runtime_kernel;
    runtime_kernel.readKernel(op_index, runtime_context);

    input = runtime_kernel.inputs[inputTensorIdx];
    output = runtime_kernel.outputs[outputTensorIdx];

    assert(input != nullptr);
    assert(output != nullptr);

    status = runtime_kernel.getDataFromStorage(op_index, runtime_storage, runtime_context);
    if (status != Ok)
      return status;

    input_data = runtime_kernel.inputs_data[inputTensorIdx];
    output_data = runtime_kernel.outputs_data[outputTensorIdx];
  }

  assert(input_data != nullptr);
  assert(output_data != nullptr);

  assert(output->type() == circle::TensorType_FLOAT32);

  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_INT8:
    {
      assert(input->quantization() != nullptr);
      assert(input->quantization()->scale() != nullptr and
             input->quantization()->scale()->size() == 1);
      assert(input->quantization()->zero_point() != nullptr and
             input->quantization()->zero_point()->size() == 1);
      core::QuantizationParams params{};
      params.zero_point = input->quantization()->zero_point()->operator[](0);
      params.scale = input->quantization()->scale()->operator[](0);

      status = pal::Dequantize(params, core::OMRuntimeShape(input).flatSize(),
                               core::utils::castInputData<int8_t>(input_data),
                               core::utils::castOutputData<float>(output_data));
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
