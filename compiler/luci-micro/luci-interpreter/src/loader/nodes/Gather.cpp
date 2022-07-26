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

std::unique_ptr<Kernel> build_kernel_CircleGather(const luci::CircleNode *circle_node,
                                                  KernelBuilderHelper &helper)
{
  const auto *node = loco::must_cast<const luci::CircleGather *>(circle_node);
  assert(node->arity() == 2);

  const Tensor *params = helper.getInputTensor(node->params());
  const Tensor *indices = helper.getInputTensor(node->indices());
  Tensor *output = helper.getOutputTensor(node);

  GatherParams gparams{};
  gparams.axis = node->axis();
  // TODO support batch_dims
  gparams.batch_dims = 0;

  return std::make_unique<kernels::Gather>(params, indices, output, gparams);
}

} // namespace luci_interpreter
