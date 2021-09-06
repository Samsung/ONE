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

#include "L2Pool2D.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleL2Pool2D(const luci::CircleNode *circle_node,
                                                    KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleL2Pool2D *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 1);

  const Tensor *input = helper.getInputTensor(node->value());
  Tensor *output = helper.getOutputTensor(node);

  Pool2DParams params{};
  params.padding = node->padding();
  params.filter_height = node->filter()->h();
  params.filter_width = node->filter()->w();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::L2Pool2D>(input, output, params);
}

} // namespace luci_interpreter
