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

#include "execute/OMKernelExecutionBuilder.h"
#include "execute/OMRuntimeKernel.h"

#include "PALSoftmax.h"

using namespace onert_micro;
using namespace onert_micro::execute;

namespace
{

constexpr uint32_t inputTensorIdx = 0;
constexpr uint32_t outputTensorIdx = 0;

} // namespace

// NOTE: doesnt currently support dynamic shapes
OMStatus onert_micro::execute::execute_kernel_CircleSoftmax(const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &runtime_context = execute_args.runtime_context;
  core::OMRuntimeStorage &runtime_storage = execute_args.runtime_storage;
  uint16_t op_index = execute_args.kernel_index;

  const circle::Tensor *input = nullptr;
  const circle::Tensor *output = nullptr;

  uint8_t *input_data = nullptr;
  uint8_t *output_data = nullptr;

  OMStatus status = Ok;

  const circle::SoftmaxOptions *options;
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

    options = runtime_kernel.first_operator->builtin_options_as_SoftmaxOptions();
  }

  assert(input_data != nullptr);
  assert(output_data != nullptr);

  const float beta = options->beta();

  core::OMRuntimeShape inputs_shape(input);
  core::OMRuntimeShape outputs_shape(output);

  const auto dim_count = inputs_shape.dimensionsCount();

  const auto trailing_dim = dim_count - 1;

  int flat_size = 1;
  for (int i = 0; i < inputs_shape.dimensionsCount(); ++i)
  {
    flat_size *= (i == trailing_dim) ? 1 : inputs_shape.dims(i);
  }

  core::SoftmaxParams params{};
  params.beta = beta;
  params.num_rows = flat_size;
  params.row_size = std::min(inputs_shape.dims(trailing_dim), outputs_shape.dims(trailing_dim));

  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
    {

      status = pal::Softmax(params, core::utils::castInputData<float>(input_data),
                            core::utils::castOutputData<float>(output_data));
    }
    break;
#endif // DIS_FLOAT
#ifndef DIS_QUANT
    case circle::TensorType_INT8:
    {
      assert(output->type() == circle::TensorType_INT8);
      if (output->type() != circle::TensorType_INT8)
        return UnsupportedType;

      assert(input->quantization() != nullptr and output->quantization() != nullptr);
      assert(input->quantization()->scale() != nullptr and
             output->quantization()->scale() != nullptr);
      assert(input->quantization()->zero_point() != nullptr and
             output->quantization()->zero_point() != nullptr);
      assert(input->quantization()->scale()->size() == 1 and
             output->quantization()->scale()->size() == 1);
      assert(input->quantization()->zero_point()->size() == 1 and
             output->quantization()->zero_point()->size() == 1);

      params.output_scale = output->quantization()->scale()->operator[](0);
      params.input_scale = input->quantization()->scale()->operator[](0);
      params.output_zp = output->quantization()->zero_point()->operator[](0);
      params.input_zp = input->quantization()->zero_point()->operator[](0);

      status = pal::Softmax(params, core::utils::castInputData<int8_t>(input_data),
                            core::utils::castOutputData<int8_t>(output_data));
    }
    break;
#endif // DIS_QUANT
    default:
    {
      status = UnsupportedType;
      assert(false && "Unsupported type.");
    }
  }

  return status;
}
