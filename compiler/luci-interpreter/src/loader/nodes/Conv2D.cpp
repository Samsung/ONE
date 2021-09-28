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

#include "kernels/Conv2D.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleConv2D(const luci::CircleNode *circle_node,
                                                  KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleConv2D *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 3);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *filter = helper.getInputTensor(node->filter());
  const Tensor *bias = helper.getInputTensor(node->bias());
  Tensor *output = helper.getOutputTensor(node);

  auto im2col =
    std::make_unique<Tensor>(input->element_type(), Shape({}), AffineQuantization{}, "");
  im2col->set_observable(false);
  im2col->set_data_buffer(nullptr);
  Tensor *tmp = helper.getRuntimeGraph(node->graph())->addTensor(std::move(im2col));

  Conv2DParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.dilation_height_factor = node->dilation()->h();
  params.dilation_width_factor = node->dilation()->w();
  params.activation = node->fusedActivationFunction();

  return std::make_unique<kernels::Conv2D>(input, filter, bias, output, tmp, params);
}

} // namespace luci_interpreter
