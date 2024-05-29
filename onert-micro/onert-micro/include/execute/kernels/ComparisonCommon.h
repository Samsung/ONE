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

#ifndef ONERT_MICRO_EXECUTE_KERNELS_COMPARISONCOMMON_H
#define ONERT_MICRO_EXECUTE_KERNELS_COMPARISONCOMMON_H

#include "OMStatus.h"

#include "core/OMUtils.h"
#include "core/OMKernelData.h"
#include "execute/OMRuntimeKernel.h"

#include "PALComparisons.h"

namespace onert_micro
{
namespace execute
{

namespace
{

constexpr uint32_t input1TensorIdx = 0;
constexpr uint32_t input2TensorIdx = 1;
constexpr uint32_t outputTensorIdx = 0;

} // namespace

template <typename T> void evalComparisonGeneric(OMRuntimeKernel *runtime_kernel, bool F(T, T))
{
  const circle::Tensor *input1 = nullptr;
  const circle::Tensor *input2 = nullptr;
  const circle::Tensor *output = nullptr;

  uint8_t *input_data1 = nullptr;
  uint8_t *input_data2 = nullptr;
  uint8_t *output_data = nullptr;

  input1 = runtime_kernel->inputs[input1TensorIdx];
  input2 = runtime_kernel->inputs[input2TensorIdx];
  output = runtime_kernel->outputs[outputTensorIdx];

  assert(input1 != nullptr);
  assert(input2 != nullptr);
  assert(output != nullptr);

  input_data1 = runtime_kernel->inputs_data[input1TensorIdx];
  input_data2 = runtime_kernel->inputs_data[input2TensorIdx];
  output_data = runtime_kernel->outputs_data[outputTensorIdx];

  assert(input_data1 != nullptr);
  assert(input_data2 != nullptr);
  assert(output_data != nullptr);

  core::OMRuntimeShape input1_shape(input1);
  core::OMRuntimeShape input2_shape(input2);
  core::OMRuntimeShape output_shape(output);

  auto cast_input_data1 = core::utils::castInputData<T>(input_data1);

  assert(cast_input_data1 != nullptr);

  auto cast_input_data2 = core::utils::castInputData<T>(input_data2);

  assert(cast_input_data2 != nullptr);

  auto cast_output_data = core::utils::castOutputData<bool>(output_data);

  onert_micro::core::ComparisonParams op_params;
  op_params.is_broadcast = input1_shape.flatSize() != input2_shape.flatSize();

  if (op_params.is_broadcast)
  {
    onert_micro::execute::pal::BroadcastComparison4DSlowNoScaling<T>(
      op_params, input1_shape, cast_input_data1, input2_shape, cast_input_data2, output_shape,
      cast_output_data, F);
  }
  else
  {
    const int64_t flat_size = input1_shape.flatSize();
    onert_micro::execute::pal::ComparisonNoScaling<T>(flat_size, cast_input_data1, cast_input_data2,
                                                      cast_output_data, F);
  }
}

} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_KERNELS_COMPARISONCOMMON_H
