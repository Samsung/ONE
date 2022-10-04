/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Gather.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleGather(std::vector<const Tensor *> &&inputs,
                                                  std::vector<Tensor *> &&outputs,
                                                  const uint32_t op_index, KernelBuilder &builder)
{
  assert(inputs.size() == 2);
  const Tensor *params = inputs.at(0);
  const Tensor *indices = inputs.at(1);
  Tensor *output = outputs.at(0);

  circle::OperatorT oper_t;
  builder.get_circle_reader()->operators()[op_index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsGatherOptions();

  GatherParams gparams{};
  gparams.axis = options->axis;
  // TODO support batch_dims
  gparams.batch_dims = 0;

  return std::make_unique<kernels::Gather>(params, indices, output, gparams);
}

} // namespace luci_interpreter
