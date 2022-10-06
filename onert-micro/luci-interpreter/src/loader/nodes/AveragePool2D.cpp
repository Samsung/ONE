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

#include "kernels/AveragePool2D.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleAveragePool2D(std::vector<const Tensor *> &&inputs,
                                                         std::vector<Tensor *> &&outputs,
                                                         const uint32_t op_index,
                                                         KernelBuilder &builder)
{
  assert(inputs.size() == 1);

  const Tensor *input = inputs.at(0);
  Tensor *output = outputs.at(0);

  circle::OperatorT oper_t;
  builder.get_circle_reader()->operators()[op_index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsPool2DOptions();

  Pool2DParams params{};
  params.padding = luci_padding(options->padding);
  params.filter_height = options->filter_height;
  params.filter_width = options->filter_width;
  params.stride_height = options->stride_h;
  params.stride_width = options->stride_w;
  params.activation = luci_actfunc(options->fused_activation_function);

  // It is unknown what data will be stored in scratchpad tensor,
  // using UINT8 as a most general option
  auto scratchpad = std::make_unique<Tensor>(DataType::U8, Shape({}), nullptr);
  scratchpad->set_data_buffer(nullptr);
  // TODO move tensors offset initialization to one place
  // TODO handle with static manager
  Tensor *tmp = builder.get_runtime_graph()->addTensor(std::move(scratchpad));

  return std::make_unique<kernels::AveragePool2D>(input, output, tmp, params);
}

} // namespace luci_interpreter
