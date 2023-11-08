/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/CumSum.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleCumSum(const luci::CircleNode *circle_node,
                                                  KernelBuilderHelper &helper)
{
  const auto *node = loco::must_cast<const luci::CircleCumSum *>(circle_node);
  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *axis = helper.getInputTensor(node->axis());
  Tensor *output = helper.getOutputTensor(node);

  CumSumParams params{};
  params.exclusive = node->exclusive();
  params.reverse = node->reverse();

  return std::make_unique<kernels::CumSum>(input, axis, output, params);
}

} // namespace luci_interpreter
