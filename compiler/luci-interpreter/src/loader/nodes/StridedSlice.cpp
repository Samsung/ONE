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

#include "StridedSlice.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleStridedSlice(const luci::CircleNode *circle_node, KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleStridedSlice *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 4);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *begin = helper.getInputTensor(node->begin());
  const Tensor *end = helper.getInputTensor(node->end());
  const Tensor *strides = helper.getInputTensor(node->strides());

  Tensor *output = helper.getOutputTensor(node);

  StridedSliceParams params{};
  params.begin_mask = node->begin_mask();
  params.ellipsis_mask = node->ellipsis_mask();
  params.end_mask = node->end_mask();
  params.new_axis_mask = node->new_axis_mask();
  params.shrink_axis_mask = node->shrink_axis_mask();

  return std::make_unique<kernels::StridedSlice>(input, begin, end, strides, output, params);
}

} // namespace luci_interpreter
