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
#include "execute/OMUtils.h"

#include "PALLogicalCommon.h"

using namespace onert_micro::core;

namespace onert_micro::execute
{

OMStatus execute_kernel_CircleLogicalOr(const OMExecuteArgs &execute_args)
{
  const circle::Tensor *input1;
  const circle::Tensor *input2;
  const circle::Tensor *output;

  OMRuntimeKernel rt_kernel;

  TISOHeader(execute_args, &input1, &input2, &output, &rt_kernel);

  auto input1_data = utils::castInputData<bool>(rt_kernel.inputs_data[0]);
  auto input2_data = utils::castInputData<bool>(rt_kernel.inputs_data[1]);
  auto output_data = utils::castOutputData<bool>(rt_kernel.outputs_data[0]);

  const int flat_size = OMRuntimeShape(input1).flatSize();

  return pal::LogicalCommon<pal::LogicalOrFn>(flat_size, input1_data, input2_data, output_data);
}

} // namespace onert_micro::execute
