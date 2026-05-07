/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "core/OMUtils.h"
#include "import/OMUtils.h"

using namespace onert_micro;
using namespace onert_micro::core::utils;

namespace onert_micro::import
{

OMStatus configure_kernel_CircleCumSum(const OMConfigureArgs &config_args)
{
  constexpr static uint32_t axisTensorIdx = 1;
  constexpr static uint32_t inputTensorIdx = 0;
  constexpr static uint32_t outputTensorIdx = 0;

  execute::OMRuntimeKernel rt_kernel;
  core::OMRuntimeContext &rt_context = config_args.runtime_context;

  OMStatus status = rt_kernel.readKernel(config_args.kernel_index, rt_context);

  if (status != Ok)
    return status;

  const circle::Tensor *axis = rt_kernel.inputs[axisTensorIdx];
  const circle::Tensor *input = rt_kernel.inputs[inputTensorIdx];
  const circle::Tensor *output = rt_kernel.outputs[outputTensorIdx];

  assert(axis != nullptr);
  assert(input != nullptr);
  assert(output != nullptr);

  core::OMRuntimeShape axis_shape(axis);
  core::OMRuntimeShape input_shape(input);

  // clang-format off

  std::array<bool, 4> conditions = {
    input->type() == output->type(),
    input_shape.dimensionsCount() > 1,
    axis->type() == circle::TensorType_INT32,
    axis_shape.isScalar()
  };

  bool status_check = std::all_of(conditions.cbegin(), conditions.cend(), [](bool condition)
  {
    return checkCondition(condition) == Ok;
  });

  // clang-format on

  if (!status_check)
    return FailedCheckCondition;

  return Ok;
}

} // namespace onert_micro::import
