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

#include "kernels/TransposeConv.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleTransposeConv(const luci::CircleNode *circle_node,
                                                         KernelBuilderHelper &helper)
{
  const auto *node = loco::must_cast<const luci::CircleTransposeConv *>(circle_node);
  assert(node->arity() == 4);

  const Tensor *input_sizes = helper.getInputTensor(node->inputSizes());
  const Tensor *filter = helper.getInputTensor(node->filter());
  const Tensor *out_backprop = helper.getInputTensor(node->outBackprop());
  const Tensor *bias = helper.getOptionalInputTensor(node->bias());

  Tensor *output = helper.getOutputTensor(node);

  DataType scratch_data_type =
    helper.getInputTensor(node)->element_type() == DataType::S16 ? DataType::S64 : DataType::S32;

  auto scratch_tensor =
    std::make_unique<Tensor>(scratch_data_type, Shape({}), AffineQuantization{}, "");
  scratch_tensor->set_observable(false);
  scratch_tensor->set_data_buffer(nullptr);
  Tensor *tmp = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratch_tensor));

  TransposeConvParams params{};
  params.padding = node->padding();
  params.stride_height = node->stride()->h();
  params.stride_width = node->stride()->w();
  params.activation = node->fusedActivationFunction();

  // TODO support activation
  if (params.activation != luci::FusedActFunc::NONE)
  {
    throw std::runtime_error("Unsupported activation of TransposeConv");
  }

  return std::make_unique<kernels::TransposeConv>(input_sizes, filter, out_backprop, bias, output,
                                                  tmp, params);
}

} // namespace luci_interpreter
