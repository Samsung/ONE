/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/BatchMatMul.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel>
build_kernel_CircleBatchMatMul(std::vector<std::pair<const Tensor *, int32_t>> &inputs,
                               std::vector<std::pair<Tensor *, int32_t>> &outputs,
                               const uint32_t op_index, KernelBuilder &builder)
{
  assert(inputs.size() == 2);

  const Tensor *lhs = inputs.at(0).first;
  const Tensor *rhs = inputs.at(1).first;
  Tensor *output = outputs.at(0).first;

  auto lhs_scratchpad = std::make_unique<Tensor>(lhs->element_type(), Shape({}), nullptr);
  lhs_scratchpad->set_data_buffer(nullptr);
  auto rhs_scratchpad = std::make_unique<Tensor>(rhs->element_type(), Shape({}), nullptr);
  rhs_scratchpad->set_data_buffer(nullptr);
  // TODO move tensors offset initialization to one place
  // TODO handle with StaticManager
  Tensor *lhs_tmp = builder.get_runtime_graph()->addTensor(std::move(lhs_scratchpad));
  Tensor *rhs_tmp = builder.get_runtime_graph()->addTensor(std::move(rhs_scratchpad));

  circle::OperatorT oper_t;
  builder.get_circle_reader()->operators()[op_index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsBatchMatMulOptions();

  BatchMatMulParams params;
  params.adj_x = options->adjoint_lhs;
  params.adj_y = options->adjoint_rhs;

  return std::make_unique<kernels::BatchMatMul>(lhs, rhs, output, lhs_tmp, rhs_tmp, params);
}

} // namespace luci_interpreter
