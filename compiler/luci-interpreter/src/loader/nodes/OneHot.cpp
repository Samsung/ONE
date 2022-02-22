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

#include "kernels/OneHot.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleOneHot(const luci::CircleNode *circle_node,
                                                  KernelBuilderHelper &helper)
{
  const auto *node = loco::must_cast<const luci::CircleOneHot *>(circle_node);
  assert(node->arity() == 4);

  const Tensor *indices = helper.getInputTensor(node->indices());
  const Tensor *depth = helper.getInputTensor(node->depth());
  const Tensor *on_value = helper.getInputTensor(node->on_value());
  const Tensor *off_value = helper.getInputTensor(node->off_value());
  Tensor *output = helper.getOutputTensor(node);

  OneHotParams params{};
  params.axis = node->axis();

  return std::make_unique<kernels::OneHot>(indices, depth, on_value, off_value, output, params);
}

} // namespace luci_interpreter
