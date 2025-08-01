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

using namespace onert_micro::core;

namespace onert_micro::import
{

OMStatus configure_kernel_CircleLogicalNot(const OMConfigureArgs &config_args)
{
  const circle::Tensor *input;
  const circle::Tensor *output;

  SISOHeader(config_args, &input, &output);

  bool type_check = (input->type() == circle::TensorType_BOOL && input->type() == output->type());

  OMStatus status = utils::checkCondition(type_check);

  if (status != Ok)
    return status;

  const OMRuntimeShape input_shape(input);
  const OMRuntimeShape output_shape(output);

  return utils::checkCondition(input_shape == output_shape);
}

} // namespace onert_micro::import
