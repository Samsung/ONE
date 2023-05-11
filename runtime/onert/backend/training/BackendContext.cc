/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "BackendContext.h"

#include "TensorBuilder.h"
#include "KernelGenerator.h"

#include <backend/basic/BackendContextHelpers.h>

namespace onert
{
namespace backend
{
namespace training
{

ITensorRegistry *BackendContext::genTensors() { return basic::genTensors(*this); }

FunctionMap BackendContext::genKernels()
{
  FunctionMap ret;

  for (auto op_ind : _data.op_order)
  {
    auto fn_seq = kernel_gen->generate(op_ind);
    ret.emplace_back(op_ind, std::move(fn_seq));
  }

  basic::initConsts(*this);

  // NOTE For memory optimization, we want to free some operand data
  const_cast<ir::Graph &>(*_data.graph)
    .operands()
    .iterate([&](const ir::OperandIndex &, ir::Operand &obj) { obj.releaseData(); });

  for (auto &&it : ret)
  {
    auto &fn_seq = it.second;
    fn_seq->iterate([&](exec::IFunction &ifunc) { ifunc.prepare(); });
  }

  return ret;
}

} // namespace training
} // namespace backend
} // namespace onert
