/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Builders.h"

#include "kernels/Unpack.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel>
build_kernel_CircleUnpack(std::vector<std::pair<const Tensor *, int32_t>> &inputs,
                          std::vector<std::pair<Tensor *, int32_t>> &outputs,
                          const uint32_t op_index, KernelBuilder &builder)
{
  assert(inputs.size() == 1);

  const Tensor *input = inputs.at(0).first;
  std::vector<Tensor *> output_tensors(outputs.size());

  for (uint32_t i = 0; i < outputs.size(); ++i)
  {
    output_tensors[i] = outputs.at(i).first;
  }

  circle::OperatorT oper_t;
  builder.get_circle_reader()->operators()[op_index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsUnpackOptions();

  UnpackParams params{};
  params.axis = options->axis;

  // NOTE 'num' attribute is ignored.
  return std::make_unique<kernels::Unpack>(input, std::move(output_tensors), params);
}

} // namespace luci_interpreter
