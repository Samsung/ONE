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

#include "Reshape.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleReshape(const luci::CircleNode *circle_node,
                                                   KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleReshape *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->tensor());
  const Tensor *shape = helper.getInputTensor(node->shape());
  Tensor *output = helper.getOutputTensor(node);

  // NOTE 'newShape' attribute is ignored.
  return std::make_unique<kernels::Reshape>(input, shape, output);
}

} // namespace luci_interpreter
