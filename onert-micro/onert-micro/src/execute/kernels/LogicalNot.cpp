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

#include "PALLogicalNotCommon.h"

using namespace onert_micro::core;

namespace onert_micro::execute
{

OMStatus execute_kernel_CircleLogicalNot(const OMExecuteArgs &execute_args)
{
  const circle::Tensor *input;
  const circle::Tensor *output;

  uint8_t *input_data;
  uint8_t *output_data;

  SISOHeader(execute_args, &input, &output, &input_data, &output_data);

  auto bool_input_data = utils::castInputData<bool>(input_data);
  auto bool_output_data = utils::castOutputData<bool>(output_data);

  const int flat_size = OMRuntimeShape(input).flatSize();

  return pal::LogicalNot(flat_size, bool_input_data, bool_output_data);
}

} // namespace onert_micro::execute
