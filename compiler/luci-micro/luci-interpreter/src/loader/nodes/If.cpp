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

#include "kernels/If.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleIf(std::vector<const Tensor *> &&inputs,
                                              std::vector<Tensor *> &&outputs,
                                              const uint32_t op_index, KernelBuilder &builder)
{
  // TODO: support IF operation
  throw std::runtime_error("Not supported now");
}

} // namespace luci_interpreter
