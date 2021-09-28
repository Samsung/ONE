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

#include "kernels/ResizeBilinear.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleResizeBilinear(const luci::CircleNode *circle_node,
                                                          KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleResizeBilinear *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *size = helper.getInputTensor(node->size());
  Tensor *output = helper.getOutputTensor(node);

  ResizeBilinearParams params{};
  params.align_corners = node->align_corners();
  params.half_pixel_centers = node->half_pixel_centers();

  return std::make_unique<kernels::ResizeBilinear>(input, size, output, params);
}

} // namespace luci_interpreter
