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

std::unique_ptr<Kernel>
build_kernel_CircleTransposeConv(std::vector<std::pair<const Tensor *, int32_t>> &inputs,
                                 std::vector<std::pair<Tensor *, int32_t>> &outputs,
                                 const uint32_t op_index, KernelBuilder &builder)
{
  assert(inputs.size() == 4);

  const Tensor *input_sizes = inputs.at(0).first;
  const Tensor *filter = inputs.at(1).first;
  const Tensor *out_backprop = inputs.at(2).first;
  const Tensor *bias = inputs.at(3).first;
  Tensor *output = outputs.at(0).first;

  DataType scratch_data_type =
    input_sizes->element_type() == DataType::S16 ? DataType::S64 : DataType::S32;

  auto scratch_tensor = std::make_unique<Tensor>(scratch_data_type, Shape({}), nullptr);
  scratch_tensor->set_data_buffer(nullptr);
  Tensor *tmp = builder.get_runtime_graph()->addTensor(std::move(scratch_tensor));

  circle::OperatorT oper_t;
  builder.get_circle_reader()->operators()[op_index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsTransposeConvOptions();

  TransposeConvParams params{};
  params.padding = luci_padding(options->padding);
  params.stride_height = options->stride_h;
  params.stride_width = options->stride_w;

  return std::make_unique<kernels::TransposeConv>(input_sizes, filter, out_backprop, bias, output,
                                                  tmp, params);
}

} // namespace luci_interpreter
