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

#include "kernels/Sqrt.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel>
build_kernel_CircleSqrt(std::vector<std::pair<const Tensor *, int32_t>> &inputs,
                        std::vector<std::pair<Tensor *, int32_t>> &outputs, const uint32_t op_index,
                        KernelBuilder &builder)
{
  assert(inputs.size() == 1);
  const Tensor *input = inputs.at(0).first;
  Tensor *output = outputs.at(0).first;

  return std::make_unique<kernels::Sqrt>(input, output);
}

} // namespace luci_interpreter
