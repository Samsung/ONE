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

OMStatus configure_kernel_CircleLogicalOr(const OMConfigureArgs &config_args)
{
  const circle::Tensor *input1;
  const circle::Tensor *input2;
  const circle::Tensor *output;

  TISOHeader(config_args, &input1, &input2, &output);

  std::array<circle::TensorType, 3> types = {input1->type(), input2->type(), output->type()};

  // clang-format off

  bool type_check = std::all_of(types.cbegin(), types.cend(), [](const auto &type)
  {
    return type == circle::TensorType_BOOL;
  });

  // clang-format on

  OMStatus status = utils::checkCondition(type_check);

  if (status != Ok)
    return status;

  const OMRuntimeShape input_shape1(input1);
  const OMRuntimeShape input_shape2(input2);

  status = utils::checkCondition(input_shape1.flatSize() == input_shape2.flatSize());

  if (status != Ok)
    return status;

  return utils::checkCondition(input_shape1.dimensionsCount() == input_shape2.dimensionsCount());
}

} // namespace onert_micro::import
