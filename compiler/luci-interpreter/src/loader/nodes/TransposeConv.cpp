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

#include "TransposeConv.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleTransposeConv(const luci::CircleNode *circle_node,
                                                         KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleTransposeConv *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 4);

  const Tensor *input_sizes = helper.getInputTensor(node->inputSizes());
  const Tensor *filter = helper.getInputTensor(node->filter());
  const Tensor *out_backprop = helper.getInputTensor(node->outBackprop());
  const Tensor *bias = helper.getOptionalInputTensor(node->bias());

  Tensor *output = helper.getOutputTensor(node);

  TransposeConvParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();

  return std::make_unique<kernels::TransposeConv>(input_sizes, filter, out_backprop, bias, output,
                                                  params);
}

} // namespace luci_interpreter
