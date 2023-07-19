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

#include "backend/basic/train/TrainableBackendContextHelpers.h"
#include "exec/FunctionSequence.h"

namespace onert
{
namespace backend
{
namespace builtin
{
namespace train
{

backend::ITensorRegistry *BackendContext::genTensors()
{
  // For now, there is no need to generate tensors for forwarding.
  // builtin train backend handles 3 operators: `Permute`, `IF`, `WHILE`.
  // `Permute`: Tensor generation is not required.
  // `IF`, `WHILE`: Not supported yet
  return tensor_registry().get();
}

backend::train::ITensorRegistry *BackendContext::genTrainingTensors()
{
  // For now, there is no need to generate tensors for backwarding.
  return tensor_registry().get();
}

backend::train::FunctionMap BackendContext::genKernels()
{
  backend::train::FunctionMap ret;

  for (auto &&op_ind : _tdata->op_order)
  {
    auto tn_seq = kernel_gen->generate(op_ind);
    ret.emplace_back(op_ind, std::move(tn_seq));
  }

  trainable_graph()->operands().iterate(
    [&](const ir::OperandIndex &ind, const ir::Operand &operand) {
      if (!external_operands().contains(ind) && operand.isConstant())
      {
        throw std::runtime_error(
          "BackendContext: builtin backend does not support updatable weights yet");
      }
    });

  // TODO Enable prepare()
  // for (auto &&it : ret)
  // {
  //   auto &fn_seq = it.second;
  //   fn_seq->iterate([&](exec::IFunction &ifunc) { ifunc.prepare(); });
  // }

  return ret;
}

} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert
